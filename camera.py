"""Tapo C236 (and similar RTSP/ONVIF) network-camera capture.

Single-frame capture over RTSP for the assistant. Mirrors the module conventions
of eyes.py (OpenCV frame grab, warm-up discard, safe error strings) but targets a
networked IP camera addressed by host/port + stream path rather than a local
webcam index.

Design notes:
- The RTSP URL is built programmatically from individual env vars and is NEVER
  logged. Credentials are URL-encoded so characters like @ : / in a password do
  not break the URL. redact() scrubs credentials out of any string before it can
  reach the console or a returned error.
- capture_camera_frame() is the assistant-facing tool. It never raises: on any
  failure it returns a structured error dict so a dead camera can't take down the
  whole assistant loop.
- The client is stream-agnostic: TAPO_CAMERA_STREAM (or a per-call override)
  selects the high-res /stream1 or the low-res /stream2 path.
- PTZ lives in the optional camera_ptz.py (ONVIF); this module intentionally
  knows nothing about it so basic RTSP capture has no ONVIF dependency.
"""

import argparse
import os
import re
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import quote

import cv2
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()
console = Console()

DEFAULT_WARMUP_FRAMES = 10  # discard early frames while the decoder/exposure settles
CAPTURE_TIMEOUT_MS = 8000  # open/read timeout so a disconnected camera fails fast instead of hanging


class CameraError(RuntimeError):
    """Raised for connection/read/write failures. Message is always credential-safe."""


class CameraConfigError(CameraError):
    """Raised when required camera settings are missing."""


@dataclass(frozen=True)
class CameraConfig:
    host: str
    port: int
    username: str
    password: str
    stream: str  # "stream1" (high-res) or "stream2" (low-res); leading slash tolerated
    name: str
    capture_dir: str

    @property
    def stream_path(self):
        """The RTSP path, e.g. '/stream1'. Accepts 'stream1' or '/stream1'."""
        return "/" + self.stream.strip().lstrip("/")

    @property
    def stream_label(self):
        """Bare stream name for metadata, e.g. 'stream1'."""
        return self.stream.strip().lstrip("/")


def load_config(stream_override=None):
    """Build a CameraConfig from TAPO_CAMERA_* env vars. Does not validate — call
    validate_config() before connecting so callers control the failure path."""
    stream = stream_override or os.environ.get("TAPO_CAMERA_STREAM", "stream1")
    return CameraConfig(
        host=os.environ.get("TAPO_CAMERA_HOST", "").strip(),
        port=int(os.environ.get("TAPO_CAMERA_PORT", "554") or "554"),
        username=os.environ.get("TAPO_CAMERA_USERNAME", ""),
        password=os.environ.get("TAPO_CAMERA_PASSWORD", ""),
        stream=stream,
        name=os.environ.get("TAPO_CAMERA_NAME", "office").strip() or "office",
        capture_dir=os.environ.get("TAPO_CAMERA_CAPTURE_DIR", "artifacts/camera"),
    )


def validate_config(config):
    """Raise CameraConfigError listing any missing required settings."""
    missing = []
    if not config.host:
        missing.append("TAPO_CAMERA_HOST")
    if not config.username:
        missing.append("TAPO_CAMERA_USERNAME")
    if not config.password:
        missing.append("TAPO_CAMERA_PASSWORD")
    if missing:
        raise CameraConfigError("Missing required camera settings: " + ", ".join(missing))


def build_rtsp_url(config):
    """Construct rtsp://<encoded-user>:<encoded-pass>@host:port/streamN.

    Never log the return value — it contains credentials. safe='' forces encoding
    of reserved characters (@ : / etc.) so special-character passwords are safe.
    """
    user = quote(config.username, safe="")
    password = quote(config.password, safe="")
    return f"rtsp://{user}:{password}@{config.host}:{config.port}{config.stream_path}"


_RTSP_CRED_RE = re.compile(r"(rtsp://)[^@/\s]*@")


def redact(text, config=None):
    """Remove credentials from any string (URLs, exception messages) before it is
    logged or returned. Scrubs the generic rtsp://user:pass@ block, and — when a
    config is given — the specific username/password in raw and URL-encoded form."""
    if not text:
        return text
    result = _RTSP_CRED_RE.sub(r"\1***:***@", str(text))
    if config is not None:
        secrets = [
            config.password,
            quote(config.password, safe=""),
            config.username,
            quote(config.username, safe=""),
        ]
        for secret in secrets:
            if secret:
                result = result.replace(secret, "***")
    return result


def _apply_timeout(cap):
    """Best-effort per-stream open/read timeouts. These CAP_PROP_* constants exist
    on OpenCV >= 4.5; guard with getattr so an older build degrades gracefully."""
    for prop_name in ("CAP_PROP_OPEN_TIMEOUT_MSEC", "CAP_PROP_READ_TIMEOUT_MSEC"):
        prop = getattr(cv2, prop_name, None)
        if prop is not None:
            cap.set(prop, CAPTURE_TIMEOUT_MS)


class TapoCamera:
    """RTSP client for a single configured camera. Not thread-safe. Each public
    method opens and closes its own capture so no RTSP connection is left open
    after a one-shot request; use as a context manager to be explicit."""

    def __init__(self, config=None):
        self.config = config or load_config()
        self._cap = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _open(self):
        validate_config(self.config)
        # Force TCP + a socket timeout at the FFmpeg layer; UDP RTSP silently
        # stalls on some networks, and without stimeout a dead host hangs read().
        os.environ.setdefault(
            "OPENCV_FFMPEG_CAPTURE_OPTIONS",
            f"rtsp_transport;tcp|stimeout;{CAPTURE_TIMEOUT_MS * 1000}",
        )
        url = build_rtsp_url(self.config)
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        _apply_timeout(cap)
        if not cap.isOpened():
            cap.release()
            # url deliberately excluded from the message — it carries credentials
            raise CameraError(f"Could not open RTSP stream for camera '{self.config.name}'.")
        self._cap = cap
        return cap

    def check_connection(self):
        """True if the RTSP stream can be opened, False otherwise. Never raises."""
        try:
            self._open()
            return True
        except CameraError:
            return False
        finally:
            self.close()

    def capture_frame(self, warmup_frames=DEFAULT_WARMUP_FRAMES):
        """Grab the newest valid frame, save it as a timestamped JPEG, and return
        capture metadata. Raises CameraError on failure. Always releases the
        capture, even on error."""
        cap = self._open()
        try:
            frame = None
            for _ in range(max(1, warmup_frames)):
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise CameraError(f"Could not read a frame from camera '{self.config.name}'.")

            height, width = frame.shape[:2]
            os.makedirs(self.config.capture_dir, exist_ok=True)
            captured_at = datetime.now().astimezone()
            filename = f"{self.config.name}_{captured_at.strftime('%Y%m%d_%H%M%S')}.jpg"
            image_path = os.path.join(self.config.capture_dir, filename)
            if not cv2.imwrite(image_path, frame):
                raise CameraError("Failed to write captured frame to disk.")

            return {
                "image_path": image_path,
                "width": int(width),
                "height": int(height),
                "captured_at": captured_at.isoformat(timespec="seconds"),
                "stream": self.config.stream_label,
            }
        finally:
            self.close()

    def get_camera_status(self):
        """Reachability + non-secret connection info. Never raises, never exposes
        credentials."""
        return {
            "camera_name": self.config.name,
            "host": self.config.host,
            "port": self.config.port,
            "stream": self.config.stream_label,
            "reachable": self.check_connection(),
        }

    def close(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def capture_camera_frame(camera_name="office", stream=None):
    """Assistant-facing tool: capture one frame from the named camera.

    Returns a structured dict; never raises. On success `success` is True and the
    image metadata is populated; on failure `success` is False and `error` holds a
    credential-free description.
    """
    config = load_config(stream_override=stream)
    result = {
        "success": False,
        "camera_name": (camera_name or config.name),
        "image_path": None,
        "captured_at": None,
        "width": None,
        "height": None,
        "stream": config.stream_label,
        "error": None,
    }

    # Only one camera is configured; reject a mismatched name rather than silently
    # capturing the wrong device. Empty/None means "the configured camera".
    requested = (camera_name or config.name).strip().lower()
    if requested not in ("", config.name.strip().lower()):
        result["error"] = f"Unknown camera '{camera_name}'. Configured camera is '{config.name}'."
        return result

    result["camera_name"] = config.name
    try:
        with TapoCamera(config) as camera:
            result.update(success=True, error=None, **camera.capture_frame())
    except CameraError as e:
        result["error"] = redact(str(e), config)
    except Exception as e:  # never let an unexpected camera fault crash the assistant
        result["error"] = redact(f"Unexpected camera error: {e}", config)
    return result


def main():
    parser = argparse.ArgumentParser(description="Manual Tapo camera connectivity/capture test.")
    parser.add_argument("--camera", default=os.environ.get("TAPO_CAMERA_NAME", "office"),
                        help="Camera name (default: configured TAPO_CAMERA_NAME).")
    parser.add_argument("--capture", action="store_true", help="Capture and save one frame.")
    parser.add_argument("--status", action="store_true", help="Print reachability/status only.")
    parser.add_argument("--low-res", action="store_true", help="Use stream2 (low-res) instead of stream1.")
    args = parser.parse_args()

    stream = "stream2" if args.low_res else None

    if args.status or not args.capture:
        status = TapoCamera(load_config(stream_override=stream)).get_camera_status()
        console.print(f"[cyan]status: {status}[/cyan]")

    if args.capture:
        result = capture_camera_frame(camera_name=args.camera, stream=stream)
        if result["success"]:
            console.print(
                f"[green]captured -> {result['image_path']} "
                f"({result['width']}x{result['height']}, {result['stream']})[/green]"
            )
        else:
            console.print(f"[red]capture failed: {result['error']}[/red]")


if __name__ == "__main__":
    main()
