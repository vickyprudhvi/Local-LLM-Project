"""Optional ONVIF pan/tilt/zoom control for the Tapo C236.

Kept separate from camera.py on purpose: basic RTSP capture must have no ONVIF
dependency. This module is only usable when ONVIF is explicitly enabled via env
and the optional `onvif-zeep` package is installed (`pip install onvif-zeep`);
`onvif` is imported lazily so importing this module never fails.

The ONVIF service port is NOT the RTSP port — it is configured independently via
TAPO_ONVIF_PORT. Nothing here moves the camera automatically, and none of it runs
during the test suite.

Enable with:
    TAPO_ONVIF_ENABLED=true
    TAPO_ONVIF_PORT=2020        # camera's ONVIF port, not 554
    TAPO_ONVIF_USERNAME=...
    TAPO_ONVIF_PASSWORD=...
"""

import os
import time
from dataclasses import dataclass
from datetime import datetime

import cv2
from dotenv import load_dotenv

from camera import CameraError, TapoCamera, load_config, redact

load_dotenv()

SCAN_PAN_VELOCITY = 0.5  # ContinuousMove x-velocity (-1.0..1.0) used for the pan sweep
SCAN_TILT_VELOCITY = 0.5  # ContinuousMove y-velocity; +y tilts up, -y tilts down (confirmed empirically on the C236)
SCAN_STEP_SECONDS = 1.2  # duration of a center<->edge pan move; the far-side leg (edge->opposite edge) uses 2x this
SCAN_TILT_STEP_SECONDS = 1.2  # same, for the tilt axis
SCAN_SETTLE_SECONDS = 0.5  # pause after a move stops, before capturing, so the frame isn't motion-blurred


class OnvifDisabledError(CameraError):
    """Raised when PTZ is used but ONVIF is not enabled/configured."""


class OnvifDependencyError(CameraError):
    """Raised when onvif-zeep is not installed."""


@dataclass(frozen=True)
class OnvifConfig:
    enabled: bool
    host: str
    port: int
    username: str
    password: str


def load_onvif_config():
    """ONVIF settings. Falls back to the RTSP host and camera credentials when the
    ONVIF-specific ones are not set, but keeps the port independent of RTSP.

    Uses `os.environ.get(key) or default` rather than `os.environ.get(key,
    default)` throughout: .env commonly defines these keys but leaves them blank
    (see .env.example), and the two-arg form of .get() only falls back when the
    key is entirely absent, not when it's present-but-empty.
    """
    cam = load_config()
    enabled = (os.environ.get("TAPO_ONVIF_ENABLED") or "false").strip().lower() in ("1", "true", "yes", "on")
    return OnvifConfig(
        enabled=enabled,
        host=(os.environ.get("TAPO_ONVIF_HOST") or cam.host).strip(),
        port=int(os.environ.get("TAPO_ONVIF_PORT") or "0"),
        username=os.environ.get("TAPO_ONVIF_USERNAME") or cam.username,
        password=os.environ.get("TAPO_ONVIF_PASSWORD") or cam.password,
    )


def onvif_available():
    """True only if ONVIF is enabled and minimally configured. Cheap, no network."""
    cfg = load_onvif_config()
    return cfg.enabled and bool(cfg.host) and cfg.port > 0 and bool(cfg.username)


class OnvifPTZ:
    """PTZ controller. Constructing it is cheap; the ONVIF connection is opened
    lazily on first use so nothing touches the network at import/instantiation."""

    def __init__(self, config=None):
        self.config = config or load_onvif_config()
        self._camera = None
        self._ptz = None
        self._media_token = None

    def _connect(self):
        if self._ptz is not None:
            return
        if not self.config.enabled:
            raise OnvifDisabledError("ONVIF is disabled. Set TAPO_ONVIF_ENABLED=true and TAPO_ONVIF_PORT.")
        if not (self.config.host and self.config.port and self.config.username):
            raise OnvifDisabledError("ONVIF is enabled but not fully configured (host/port/username).")
        try:
            from onvif import ONVIFCamera  # lazy: optional dependency
        except ImportError as e:
            raise OnvifDependencyError("onvif-zeep is not installed. Run: pip install onvif-zeep") from e

        self._camera = ONVIFCamera(self.config.host, self.config.port, self.config.username, self.config.password)
        media = self._camera.create_media_service()
        self._ptz = self._camera.create_ptz_service()
        self._media_token = media.GetProfiles()[0].token

    def get_ptz_status(self):
        self._connect()
        return self._ptz.GetStatus({"ProfileToken": self._media_token})

    def move_continuous(self, pan=0.0, tilt=0.0, duration=0.5):
        """Move at the given pan/tilt velocity (-1.0..1.0) for `duration` seconds,
        then stop. Blocks for `duration`."""
        self._connect()
        request = self._ptz.create_type("ContinuousMove")
        request.ProfileToken = self._media_token
        request.Velocity = {"PanTilt": {"x": pan, "y": tilt}}
        self._ptz.ContinuousMove(request)
        time.sleep(duration)
        self.stop()

    def stop(self):
        self._connect()
        self._ptz.Stop({"ProfileToken": self._media_token, "PanTilt": True, "Zoom": True})

    def go_to_preset(self, preset_token):
        self._connect()
        self._ptz.GotoPreset({"ProfileToken": self._media_token, "PresetToken": preset_token})

    def list_presets(self):
        self._connect()
        return self._ptz.GetPresets({"ProfileToken": self._media_token})

    def close(self):
        # zeep clients hold no persistent socket; drop references so GC can reclaim.
        self._ptz = None
        self._camera = None
        self._media_token = None


# direction to move (pan/tilt velocity) to get back to center from each offset position
_RECENTER_PAN = {"left": SCAN_PAN_VELOCITY, "right": -SCAN_PAN_VELOCITY}
_RECENTER_TILT = {"up": -SCAN_TILT_VELOCITY, "down": SCAN_TILT_VELOCITY}


def _best_effort_recenter(ptz, current_pan, current_tilt):
    """Try to return the camera to center on both axes and stop all motion,
    swallowing any error — this runs in a `finally`, so it must never raise or
    mask the real failure. current_pan/current_tilt are each "center" or an
    offset label ("left"/"right" or "up"/"down")."""
    try:
        recenter_pan = _RECENTER_PAN.get(current_pan)
        if recenter_pan is not None:
            ptz.move_continuous(pan=recenter_pan, tilt=0.0, duration=SCAN_STEP_SECONDS)
    except Exception:
        pass
    try:
        recenter_tilt = _RECENTER_TILT.get(current_tilt)
        if recenter_tilt is not None:
            ptz.move_continuous(pan=0.0, tilt=recenter_tilt, duration=SCAN_TILT_STEP_SECONDS)
    except Exception:
        pass
    try:
        ptz.stop()
    except Exception:
        pass


def _stitch_panorama(image_paths, output_path):
    """Merge captured frames into one panorama with OpenCV's Stitcher. Returns
    output_path on success, None on any failure (insufficient overlap/features,
    a plus-shaped grid's opposite arms not overlapping each other, etc.) — the
    caller falls back to describing each frame separately when this returns None."""
    images = [cv2.imread(p) for p in image_paths]
    if any(img is None for img in images):
        return None

    stitcher = cv2.Stitcher.create() if hasattr(cv2, "Stitcher") else cv2.Stitcher_create()
    status, panorama = stitcher.stitch(images)
    if status != cv2.Stitcher_OK or panorama is None:
        return None

    if not cv2.imwrite(output_path, panorama):
        return None
    return output_path


def scan_room(camera_name=None):
    """Sweep the room in a plus shape — center, then tilt up/down, then pan
    left/right, returning to center after each axis — capturing a still frame at
    each of the 5 stops. Physically moves the camera; never called during tests.

    Attempts to stitch the 5 frames into one panorama (see _stitch_panorama);
    if that succeeds, callers get a single merged image to describe (the
    simplest and most accurate path). If stitching fails, `panorama_path` is
    None and the caller falls back to describing each frame separately and
    synthesizing one summary, the same per-image-then-synthesize pattern
    plant_watcher.py uses, since the vision model is most reliable on one image
    at a time.

    Returns a structured dict; never raises:
      {"success": bool, "camera_name": str,
       "images": [{"position", "image_path", "width", "height", "captured_at", "stream"}, ...],
       "panorama_path": str or None, "error": str or None}
    """
    cam_config = load_config()
    result = {
        "success": False,
        "camera_name": camera_name or cam_config.name,
        "images": [],
        "panorama_path": None,
        "error": None,
    }

    if not onvif_available():
        result["error"] = "ONVIF PTZ is not configured. Set TAPO_ONVIF_ENABLED=true and TAPO_ONVIF_PORT."
        return result

    camera_client = TapoCamera(cam_config)
    ptz = OnvifPTZ()
    current_pan, current_tilt = "center", "center"
    try:
        frame = camera_client.capture_frame()
        frame["position"] = "center"
        result["images"].append(frame)

        # vertical sweep: center -> up -> down -> back to center
        ptz.move_continuous(pan=0.0, tilt=SCAN_TILT_VELOCITY, duration=SCAN_TILT_STEP_SECONDS)
        current_tilt = "up"
        time.sleep(SCAN_SETTLE_SECONDS)
        frame = camera_client.capture_frame()
        frame["position"] = "up"
        result["images"].append(frame)

        ptz.move_continuous(pan=0.0, tilt=-SCAN_TILT_VELOCITY, duration=SCAN_TILT_STEP_SECONDS * 2)
        current_tilt = "down"
        time.sleep(SCAN_SETTLE_SECONDS)
        frame = camera_client.capture_frame()
        frame["position"] = "down"
        result["images"].append(frame)

        ptz.move_continuous(pan=0.0, tilt=SCAN_TILT_VELOCITY, duration=SCAN_TILT_STEP_SECONDS)
        current_tilt = "center"

        # horizontal sweep: center -> left -> right -> back to center
        ptz.move_continuous(pan=-SCAN_PAN_VELOCITY, tilt=0.0, duration=SCAN_STEP_SECONDS)
        current_pan = "left"
        time.sleep(SCAN_SETTLE_SECONDS)
        frame = camera_client.capture_frame()
        frame["position"] = "left"
        result["images"].append(frame)

        ptz.move_continuous(pan=SCAN_PAN_VELOCITY, tilt=0.0, duration=SCAN_STEP_SECONDS * 2)
        current_pan = "right"
        time.sleep(SCAN_SETTLE_SECONDS)
        frame = camera_client.capture_frame()
        frame["position"] = "right"
        result["images"].append(frame)

        ptz.move_continuous(pan=-SCAN_PAN_VELOCITY, tilt=0.0, duration=SCAN_STEP_SECONDS)
        current_pan = "center"

        result["success"] = True
    except CameraError as e:
        result["error"] = redact(str(e), cam_config)
    except Exception as e:
        result["error"] = redact(f"Unexpected error during room scan: {e}", cam_config)
    finally:
        _best_effort_recenter(ptz, current_pan, current_tilt)
        ptz.close()

    if result["success"] and len(result["images"]) == 5:
        panorama_name = f"{cam_config.name}_panorama_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        panorama_path = os.path.join(cam_config.capture_dir, panorama_name)
        result["panorama_path"] = _stitch_panorama(
            [img["image_path"] for img in result["images"]], panorama_path
        )

    return result


def main():
    """Manual live test: `python -m camera_ptz --scan-room`. Physically moves the
    camera. Never prints credentials."""
    import argparse

    from rich.console import Console

    console = Console()
    parser = argparse.ArgumentParser(description="Manual ONVIF PTZ test.")
    parser.add_argument("--scan-room", action="store_true", help="Run one live left/center/right room sweep.")
    parser.add_argument("--status", action="store_true", help="Print ONVIF PTZ status only (no movement).")
    args = parser.parse_args()

    if args.status:
        try:
            console.print(f"[cyan]ptz status: {OnvifPTZ().get_ptz_status()}[/cyan]")
        except CameraError as e:
            console.print(f"[red]{e}[/red]")
        return

    if args.scan_room:
        result = scan_room()
        if not result["success"]:
            console.print(f"[red]scan failed: {result['error']}[/red]")
            return
        for image in result["images"]:
            console.print(f"[green]{image['position']}: {image['image_path']} "
                          f"({image['width']}x{image['height']})[/green]")
        if result["panorama_path"]:
            console.print(f"[green]panorama: {result['panorama_path']}[/green]")
        else:
            console.print("[yellow]panorama stitching failed — fall back to per-image description[/yellow]")


if __name__ == "__main__":
    main()
