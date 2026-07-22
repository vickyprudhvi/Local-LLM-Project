"""Unit tests for camera.py and the camera tool's router/registry wiring.

No test connects to a real camera: cv2.VideoCapture is always mocked. cv2.imwrite
runs for real against a temp dir so directory-creation and file-save behavior are
exercised end to end.
"""

import json
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import camera
from router import TOOLS, _TOOL_NAME_MAP, route_and_answer


def _config(tmp_path, **overrides):
    defaults = dict(
        host="10.0.0.65",
        port=554,
        username="user",
        password="pass",
        stream="stream1",
        name="office",
        capture_dir=str(tmp_path),
    )
    defaults.update(overrides)
    return camera.CameraConfig(**defaults)


def _fake_cap(opened=True, frame=None, read_ok=True):
    cap = MagicMock()
    cap.isOpened.return_value = opened
    if frame is None:
        frame = np.zeros((48, 64, 3), dtype=np.uint8)  # height=48, width=64
    cap.read.return_value = (read_ok, frame if read_ok else None)
    return cap


# --- configuration validation ---

def test_validate_config_passes_when_complete(tmp_path):
    camera.validate_config(_config(tmp_path))  # should not raise


@pytest.mark.parametrize("missing", ["host", "username", "password"])
def test_validate_config_raises_when_required_missing(tmp_path, missing):
    cfg = _config(tmp_path, **{missing: ""})
    with pytest.raises(camera.CameraConfigError) as exc:
        camera.validate_config(cfg)
    assert missing.upper() in str(exc.value).upper()


# --- RTSP URL construction + encoding ---

def test_build_rtsp_url_basic_format(tmp_path):
    url = camera.build_rtsp_url(_config(tmp_path))
    assert url == "rtsp://user:pass@10.0.0.65:554/stream1"


def test_build_rtsp_url_encodes_special_characters(tmp_path):
    cfg = _config(tmp_path, username="ad@min", password="p@ss:w/rd!")
    url = camera.build_rtsp_url(cfg)
    # special chars must be percent-encoded so they don't break the URL structure
    assert "ad%40min" in url
    assert "p%40ss%3Aw%2Frd%21" in url
    # the raw password must not appear, and the host section stays intact
    assert "p@ss:w/rd!" not in url
    assert "@10.0.0.65:554/stream1" in url


def test_stream_path_tolerates_leading_slash(tmp_path):
    assert _config(tmp_path, stream="/stream2").stream_path == "/stream2"
    assert _config(tmp_path, stream="stream2").stream_path == "/stream2"


# --- credential redaction ---

def test_redact_scrubs_credentials_from_url(tmp_path):
    cfg = _config(tmp_path, password="s3cr3t")
    url = camera.build_rtsp_url(cfg)
    redacted = camera.redact(url, cfg)
    assert "s3cr3t" not in redacted
    assert "user" not in redacted
    assert redacted.startswith("rtsp://***:***@")


def test_redact_scrubs_credentials_from_exception_text(tmp_path):
    cfg = _config(tmp_path, password="s3cr3t")
    msg = f"OpenCV failed on {camera.build_rtsp_url(cfg)} while connecting"
    assert "s3cr3t" not in camera.redact(msg, cfg)


# --- successful capture (mocked stream, real file write) ---

def test_capture_frame_saves_jpeg_and_returns_metadata(tmp_path):
    cfg = _config(tmp_path)
    with patch("camera.cv2.VideoCapture", return_value=_fake_cap()):
        meta = camera.TapoCamera(cfg).capture_frame(warmup_frames=3)
    assert os.path.isfile(meta["image_path"])
    assert meta["width"] == 64 and meta["height"] == 48
    assert meta["stream"] == "stream1"
    assert meta["image_path"].endswith(".jpg")
    assert "office_" in os.path.basename(meta["image_path"])


def test_capture_camera_frame_success_shape(tmp_path):
    cfg = _config(tmp_path)
    with patch("camera.load_config", return_value=cfg), \
         patch("camera.cv2.VideoCapture", return_value=_fake_cap()):
        result = camera.capture_camera_frame(camera_name="office")
    assert result["success"] is True
    assert result["error"] is None
    assert result["camera_name"] == "office"
    assert os.path.isfile(result["image_path"])


def test_capture_creates_output_directory(tmp_path):
    target = tmp_path / "nested" / "camera"
    cfg = _config(tmp_path, capture_dir=str(target))
    assert not target.exists()
    with patch("camera.cv2.VideoCapture", return_value=_fake_cap()):
        camera.TapoCamera(cfg).capture_frame(warmup_frames=2)
    assert target.is_dir()


def test_capture_releases_capture_resource(tmp_path):
    cap = _fake_cap()
    with patch("camera.cv2.VideoCapture", return_value=cap):
        camera.TapoCamera(_config(tmp_path)).capture_frame(warmup_frames=2)
    cap.release.assert_called()


# --- failure paths return safe structured errors ---

def test_capture_camera_frame_connection_failure(tmp_path):
    cfg = _config(tmp_path)
    with patch("camera.load_config", return_value=cfg), \
         patch("camera.cv2.VideoCapture", return_value=_fake_cap(opened=False)):
        result = camera.capture_camera_frame()
    assert result["success"] is False
    assert result["image_path"] is None
    assert "office" in result["error"]
    assert cfg.password not in result["error"]  # never leak credentials in errors


def test_capture_camera_frame_invalid_frame(tmp_path):
    cfg = _config(tmp_path)
    with patch("camera.load_config", return_value=cfg), \
         patch("camera.cv2.VideoCapture", return_value=_fake_cap(read_ok=False)):
        result = camera.capture_camera_frame()
    assert result["success"] is False
    assert "read a frame" in result["error"]


def test_capture_camera_frame_missing_config_is_safe(tmp_path):
    cfg = _config(tmp_path, password="")  # missing required credential
    with patch("camera.load_config", return_value=cfg):
        result = camera.capture_camera_frame()
    assert result["success"] is False
    assert "TAPO_CAMERA_PASSWORD" in result["error"]


def test_capture_camera_frame_unknown_name(tmp_path):
    cfg = _config(tmp_path, name="office")
    with patch("camera.load_config", return_value=cfg):
        result = camera.capture_camera_frame(camera_name="garage")
    assert result["success"] is False
    assert "Unknown camera" in result["error"]


def test_check_connection_true_and_false(tmp_path):
    cfg = _config(tmp_path)
    with patch("camera.cv2.VideoCapture", return_value=_fake_cap(opened=True)):
        assert camera.TapoCamera(cfg).check_connection() is True
    with patch("camera.cv2.VideoCapture", return_value=_fake_cap(opened=False)):
        assert camera.TapoCamera(cfg).check_connection() is False


def test_get_camera_status_no_credentials_exposed(tmp_path):
    cfg = _config(tmp_path, password="topsecret")
    with patch("camera.cv2.VideoCapture", return_value=_fake_cap(opened=True)):
        status = camera.TapoCamera(cfg).get_camera_status()
    assert status["reachable"] is True
    assert status["camera_name"] == "office"
    assert "topsecret" not in json.dumps(status)


# --- tool registration + router selection ---

def test_capture_camera_tool_is_registered():
    names = [t["function"]["name"] for t in TOOLS]
    assert "capture_camera" in names
    assert _TOOL_NAME_MAP["capture_camera"] == "capture_camera"


@patch("assistant.eyes.describe_local")
@patch("assistant.camera.capture_camera_frame")
def test_dispatch_capture_camera_feeds_vision_processor(mock_capture, mock_describe):
    import assistant
    from router import RouteDecision

    mock_capture.return_value = {"success": True, "camera_name": "office",
                                 "image_path": "artifacts/camera/office_x.jpg", "error": None}
    mock_describe.return_value = ("A desk with a laptop.", {"prompt_tokens": 5})

    decision = RouteDecision(mode="tool", tool="capture_camera",
                             payload=json.dumps({"camera_name": "office"}))
    reply, metrics = assistant.dispatch(decision, "what's on my desk", "what's on my desk", [], "sys")

    mock_capture.assert_called_once_with(camera_name="office")
    mock_describe.assert_called_once_with("artifacts/camera/office_x.jpg", "what's on my desk")
    assert reply == "A desk with a laptop."


@patch("assistant.camera.capture_camera_frame")
def test_dispatch_capture_camera_failure_is_safe(mock_capture):
    import assistant
    from router import RouteDecision

    mock_capture.return_value = {"success": False, "camera_name": "office",
                                 "image_path": None, "error": "Could not open RTSP stream."}
    decision = RouteDecision(mode="tool", tool="capture_camera", payload=json.dumps({"camera_name": "office"}))
    reply, _ = assistant.dispatch(decision, "check the room", "check the room", [], "sys")
    assert "couldn't capture" in reply.lower()


# --- ONVIF PTZ gating (disabled by default; no live movement) ---

def test_onvif_disabled_by_default(monkeypatch):
    import camera_ptz
    for var in ("TAPO_ONVIF_ENABLED", "TAPO_ONVIF_PORT", "TAPO_ONVIF_USERNAME"):
        monkeypatch.delenv(var, raising=False)
    assert camera_ptz.onvif_available() is False


def test_onvif_ptz_raises_when_disabled(monkeypatch):
    import camera_ptz
    monkeypatch.setenv("TAPO_ONVIF_ENABLED", "false")
    with pytest.raises(camera_ptz.OnvifDisabledError):
        camera_ptz.OnvifPTZ().get_ptz_status()


@patch("router.requests.post")
def test_router_selects_capture_camera(mock_post):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "message": {
            "content": "",
            "tool_calls": [{"function": {"name": "capture_camera", "arguments": {"camera_name": "office"}}}],
        }
    }
    mock_post.return_value = resp
    decision = route_and_answer("look through the office camera", [])
    assert decision.mode == "tool"
    assert decision.tool == "capture_camera"
    assert json.loads(decision.payload) == {"camera_name": "office"}
