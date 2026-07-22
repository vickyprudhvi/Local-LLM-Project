"""Unit tests for camera_ptz.py's scan_room() orchestration (5-point plus-shape
sweep + panorama stitching) and its wiring into router.py / assistant.py.

No test ever calls a real ONVIF service, moves a real camera, or runs OpenCV's
real Stitcher against real images: OnvifPTZ, TapoCamera, and cv2.Stitcher.create
are always mocked. This is enforced structurally, not just by convention —
scan_room() is exercised entirely through patched collaborators.
"""

from unittest.mock import MagicMock, call, patch

import pytest

import camera_ptz
from router import TOOLS, _TOOL_NAME_MAP, route_and_answer


def _frame(position):
    return {"image_path": f"artifacts/camera/office_{position}.jpg", "width": 64,
            "height": 48, "captured_at": "2026-07-21T12:00:00", "stream": "stream1"}


def _five_frames():
    # capture order in scan_room: center, up, down, left, right
    return [_frame("center"), _frame("up"), _frame("down"), _frame("left"), _frame("right")]


@pytest.fixture
def mock_ptz():
    ptz = MagicMock()
    with patch("camera_ptz.OnvifPTZ", return_value=ptz):
        yield ptz


@pytest.fixture
def mock_camera_client():
    client = MagicMock()
    client.capture_frame.side_effect = _five_frames()
    with patch("camera_ptz.TapoCamera", return_value=client):
        yield client


@pytest.fixture
def mock_stitch_success():
    with patch("camera_ptz._stitch_panorama", return_value="artifacts/camera/office_panorama_x.jpg") as m:
        yield m


@pytest.fixture
def mock_stitch_failure():
    with patch("camera_ptz._stitch_panorama", return_value=None) as m:
        yield m


def _enable_onvif(monkeypatch):
    monkeypatch.setenv("TAPO_ONVIF_ENABLED", "true")
    monkeypatch.setenv("TAPO_ONVIF_PORT", "2020")
    monkeypatch.setenv("TAPO_CAMERA_USERNAME", "user")


# --- onvif_available() gating ---

def test_onvif_available_false_when_disabled(monkeypatch):
    monkeypatch.setenv("TAPO_ONVIF_ENABLED", "false")
    assert camera_ptz.onvif_available() is False


def test_onvif_available_true_when_fully_configured(monkeypatch):
    _enable_onvif(monkeypatch)
    assert camera_ptz.onvif_available() is True


def test_scan_room_refuses_when_onvif_disabled(monkeypatch):
    monkeypatch.setenv("TAPO_ONVIF_ENABLED", "false")
    result = camera_ptz.scan_room()
    assert result["success"] is False
    assert result["images"] == []
    assert result["panorama_path"] is None
    assert "ONVIF" in result["error"]


# --- successful 5-point sweep: correct move sequence, capture order, recentering ---

def test_scan_room_success_moves_captures_and_recenters(monkeypatch, mock_ptz, mock_camera_client, mock_stitch_success):
    _enable_onvif(monkeypatch)

    with patch("camera_ptz.time.sleep"):  # skip real settle delays in tests
        result = camera_ptz.scan_room()

    assert result["success"] is True
    assert result["error"] is None
    positions = [img["position"] for img in result["images"]]
    assert positions == ["center", "up", "down", "left", "right"]

    v, t = camera_ptz.SCAN_PAN_VELOCITY, camera_ptz.SCAN_STEP_SECONDS
    tv, tt = camera_ptz.SCAN_TILT_VELOCITY, camera_ptz.SCAN_TILT_STEP_SECONDS
    mock_ptz.move_continuous.assert_has_calls([
        call(pan=0.0, tilt=tv, duration=tt),        # center -> up
        call(pan=0.0, tilt=-tv, duration=tt * 2),   # up -> down
        call(pan=0.0, tilt=tv, duration=tt),        # down -> center (tilt recenter)
        call(pan=-v, tilt=0.0, duration=t),         # center -> left
        call(pan=v, tilt=0.0, duration=t * 2),      # left -> right
        call(pan=-v, tilt=0.0, duration=t),         # right -> center (pan recenter)
    ])
    mock_ptz.close.assert_called_once()


def test_scan_room_stitches_panorama_from_all_five_frames(monkeypatch, mock_ptz, mock_camera_client, mock_stitch_success):
    _enable_onvif(monkeypatch)
    with patch("camera_ptz.time.sleep"):
        result = camera_ptz.scan_room()
    assert result["panorama_path"] == "artifacts/camera/office_panorama_x.jpg"


def test_scan_room_panorama_none_when_stitching_fails(monkeypatch, mock_ptz, mock_camera_client, mock_stitch_failure):
    _enable_onvif(monkeypatch)
    with patch("camera_ptz.time.sleep"):
        result = camera_ptz.scan_room()
    assert result["success"] is True  # capture sweep itself still succeeded
    assert result["panorama_path"] is None


def test_scan_room_never_touches_real_ptz_class(monkeypatch, mock_ptz, mock_camera_client, mock_stitch_success):
    """Sanity check that the fixtures actually intercept the real classes."""
    _enable_onvif(monkeypatch)
    with patch("camera_ptz.time.sleep"):
        camera_ptz.scan_room()
    assert mock_ptz.move_continuous.called  # only the mock was ever driven


# --- mid-sweep failure recenters the correct axis and fails safely ---

def test_scan_room_recenters_tilt_axis_on_failure_during_vertical_sweep(monkeypatch, mock_ptz):
    _enable_onvif(monkeypatch)

    client = MagicMock()
    # center ok, up ok — then the move on to "down" blows up, leaving the camera at "up"
    client.capture_frame.side_effect = [_frame("center"), _frame("up")]
    mock_ptz.move_continuous.side_effect = [
        None,  # center -> up succeeds
        camera_ptz.CameraError("stream dropped"),  # up -> down fails
    ]
    with patch("camera_ptz.TapoCamera", return_value=client), patch("camera_ptz.time.sleep"):
        result = camera_ptz.scan_room()

    assert result["success"] is False
    assert "stream dropped" in result["error"]
    assert result["panorama_path"] is None
    tv, tt = camera_ptz.SCAN_TILT_VELOCITY, camera_ptz.SCAN_TILT_STEP_SECONDS
    # recenter (in `finally`) must move tilt back down from "up": negative velocity
    recenter_call = mock_ptz.move_continuous.call_args_list[-1]
    assert recenter_call == call(pan=0.0, tilt=-tv, duration=tt)
    mock_ptz.stop.assert_called_once()


def test_scan_room_recenters_pan_axis_on_failure_during_horizontal_sweep(monkeypatch, mock_ptz):
    _enable_onvif(monkeypatch)

    client = MagicMock()
    # vertical sweep completes fully (center, up, down, back to center-tilt), then
    # pan left succeeds, then the move on to "right" blows up
    client.capture_frame.side_effect = [_frame("center"), _frame("up"), _frame("down"), _frame("left")]
    mock_ptz.move_continuous.side_effect = [
        None,  # center -> up
        None,  # up -> down
        None,  # down -> center (tilt recenter)
        None,  # center -> left
        camera_ptz.CameraError("stream dropped"),  # left -> right fails
    ]
    with patch("camera_ptz.TapoCamera", return_value=client), patch("camera_ptz.time.sleep"):
        result = camera_ptz.scan_room()

    assert result["success"] is False
    v, t = camera_ptz.SCAN_PAN_VELOCITY, camera_ptz.SCAN_STEP_SECONDS
    # tilt is already centered, so only a pan recenter should follow; last call
    # must return from "left": positive velocity
    recenter_call = mock_ptz.move_continuous.call_args_list[-1]
    assert recenter_call == call(pan=v, tilt=0.0, duration=t)


def test_scan_room_recenter_failure_does_not_raise(monkeypatch, mock_ptz, mock_camera_client, mock_stitch_success):
    _enable_onvif(monkeypatch)
    mock_ptz.stop.side_effect = RuntimeError("network blip")

    with patch("camera_ptz.time.sleep"):
        result = camera_ptz.scan_room()  # must not raise even though stop() fails

    assert result["success"] is True  # the sweep itself still completed


def test_scan_room_credentials_never_leak_in_error(monkeypatch, mock_ptz):
    _enable_onvif(monkeypatch)
    monkeypatch.setenv("TAPO_CAMERA_PASSWORD", "s3cr3t")

    client = MagicMock()
    client.capture_frame.side_effect = RuntimeError("rtsp://user:s3cr3t@10.0.0.65:554/stream1 timed out")
    with patch("camera_ptz.TapoCamera", return_value=client), patch("camera_ptz.time.sleep"):
        result = camera_ptz.scan_room()

    assert "s3cr3t" not in result["error"]


# --- panorama stitching helper ---

def test_stitch_panorama_returns_none_on_unreadable_image(tmp_path):
    bad_path = str(tmp_path / "not_an_image.jpg")
    with open(bad_path, "w") as f:
        f.write("not a jpeg")
    good_path = str(tmp_path / "also_missing.jpg")  # doesn't exist -> cv2.imread returns None
    result = camera_ptz._stitch_panorama([bad_path, good_path], str(tmp_path / "out.jpg"))
    assert result is None


def test_stitch_panorama_returns_none_when_stitcher_reports_failure(tmp_path):
    import numpy as np
    import cv2

    img_path = str(tmp_path / "frame.jpg")
    cv2.imwrite(img_path, np.zeros((10, 10, 3), dtype="uint8"))

    fake_stitcher = MagicMock()
    fake_stitcher.stitch.return_value = (1, None)  # any non-OK status
    with patch("camera_ptz.cv2.Stitcher") as mock_stitcher_cls:
        mock_stitcher_cls.create.return_value = fake_stitcher
        result = camera_ptz._stitch_panorama([img_path, img_path], str(tmp_path / "out.jpg"))
    assert result is None


def test_stitch_panorama_returns_path_on_success(tmp_path):
    import numpy as np
    import cv2

    img_path = str(tmp_path / "frame.jpg")
    cv2.imwrite(img_path, np.zeros((10, 10, 3), dtype="uint8"))
    out_path = str(tmp_path / "out.jpg")

    fake_stitcher = MagicMock()
    fake_stitcher.stitch.return_value = (cv2.Stitcher_OK, np.zeros((10, 20, 3), dtype="uint8"))
    with patch("camera_ptz.cv2.Stitcher") as mock_stitcher_cls:
        mock_stitcher_cls.create.return_value = fake_stitcher
        result = camera_ptz._stitch_panorama([img_path, img_path], out_path)
    assert result == out_path
    import os
    assert os.path.isfile(out_path)


# --- tool registration + router selection ---

def test_scan_room_tool_is_registered():
    names = [t["function"]["name"] for t in TOOLS]
    assert "scan_room" in names
    assert _TOOL_NAME_MAP["scan_room"] == "scan_room"


@patch("router.requests.post")
def test_router_selects_scan_room(mock_post):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "message": {"content": "", "tool_calls": [{"function": {"name": "scan_room", "arguments": {}}}]}
    }
    mock_post.return_value = resp
    decision = route_and_answer("look at the complete room", [])
    assert decision.mode == "tool"
    assert decision.tool == "scan_room"


# --- dispatch wiring: panorama path (single vision call) ---

@patch("assistant.eyes.describe_local")
@patch("assistant.camera_ptz.scan_room")
def test_dispatch_scan_room_uses_panorama_when_available(mock_scan, mock_describe):
    import assistant
    from router import RouteDecision

    mock_scan.return_value = {
        "success": True, "camera_name": "office",
        "images": [{"position": p, "image_path": f"{p}.jpg"} for p in ("center", "up", "down", "left", "right")],
        "panorama_path": "artifacts/camera/office_panorama_x.jpg",
        "error": None,
    }
    mock_describe.return_value = ("There's a desk with a laptop and a bookshelf behind it.", {"prompt_tokens": 20, "completion_tokens": 15})

    decision = RouteDecision(mode="tool", tool="scan_room", payload=None)
    reply, metrics = assistant.dispatch(decision, "what's on the bookshelf", "what's on the bookshelf", [], "sys")

    mock_describe.assert_called_once_with(
        "artifacts/camera/office_panorama_x.jpg", "what's on the bookshelf",
        max_side=assistant.SCAN_ROOM_VISION_MAX_SIDE,
    )
    assert reply == "There's a desk with a laptop and a bookshelf behind it."
    assert metrics == {"prompt_tokens": 20, "completion_tokens": 15}


# --- dispatch wiring: fallback path (stitching failed -> per-image + synthesize) ---

@patch("assistant.ask_local")
@patch("assistant.eyes.describe_local")
@patch("assistant.camera_ptz.scan_room")
def test_dispatch_scan_room_falls_back_when_panorama_missing(mock_scan, mock_describe, mock_ask_local):
    import assistant
    from router import RouteDecision

    mock_scan.return_value = {
        "success": True, "camera_name": "office",
        "images": [
            {"position": "center", "image_path": "c.jpg"},
            {"position": "up", "image_path": "u.jpg"},
            {"position": "down", "image_path": "d.jpg"},
            {"position": "left", "image_path": "l.jpg"},
            {"position": "right", "image_path": "r.jpg"},
        ],
        "panorama_path": None,
        "error": None,
    }
    mock_describe.side_effect = [
        ("a desk", {"prompt_tokens": 1, "completion_tokens": 1}),
        ("a ceiling fan", {"prompt_tokens": 1, "completion_tokens": 1}),
        ("a floor", {"prompt_tokens": 1, "completion_tokens": 1}),
        ("a bookshelf", {"prompt_tokens": 1, "completion_tokens": 1}),
        ("a window", {"prompt_tokens": 1, "completion_tokens": 1}),
    ]
    mock_ask_local.return_value = ("The room has a desk, a bookshelf, and a window.", {"prompt_tokens": 2, "completion_tokens": 3})

    decision = RouteDecision(mode="tool", tool="scan_room", payload=None)
    reply, metrics = assistant.dispatch(decision, "what's on the bookshelf", "what's on the bookshelf", [], "sys")

    assert mock_describe.call_count == 5
    for describe_call in mock_describe.call_args_list:
        assert describe_call.args[1] == "what's on the bookshelf"

    synth_prompt = mock_ask_local.call_args[0][0]
    assert "a bookshelf" in synth_prompt and "a ceiling fan" in synth_prompt
    synth_system_prompt = mock_ask_local.call_args.kwargs["system_prompt"]
    assert "what's on the bookshelf" in synth_system_prompt

    assert reply == "The room has a desk, a bookshelf, and a window."
    assert metrics == {"prompt_tokens": 7, "completion_tokens": 8}


@patch("assistant.camera_ptz.scan_room")
def test_dispatch_scan_room_failure_is_safe(mock_scan):
    import assistant
    from router import RouteDecision

    mock_scan.return_value = {"success": False, "camera_name": "office", "images": [],
                              "panorama_path": None, "error": "ONVIF PTZ is not configured."}
    decision = RouteDecision(mode="tool", tool="scan_room", payload=None)
    reply, _ = assistant.dispatch(decision, "look around", "look around", [], "sys")
    assert "couldn't scan" in reply.lower()
