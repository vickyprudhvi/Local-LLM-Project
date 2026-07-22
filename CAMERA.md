# Tapo C236 network camera

RTSP still-frame capture for the assistant, plus optional ONVIF pan/tilt.

## Request flow

```
user request
  → router.py            (LLM picks the `capture_camera` tool)
  → assistant.dispatch   (tool == "capture_camera")
  → camera.capture_camera_frame()   → saves JPEG, returns structured metadata
  → eyes.describe_local()           (existing vision processor — swap in describe_claude for accurate reading)
  → spoken/printed reply
```

`capture_camera` is the fixed **IP camera**. The existing `look` / `look_carefully`
tools use the laptop **webcam** and are unchanged.

## Setup

1. Copy `.env.example` to `.env` and fill in the camera credentials (`.env` is
   gitignored — never commit it):

   ```
   TAPO_CAMERA_HOST=10.0.0.65
   TAPO_CAMERA_PORT=554
   TAPO_CAMERA_USERNAME=<camera account user>
   TAPO_CAMERA_PASSWORD=<camera account password>
   TAPO_CAMERA_STREAM=stream1        # stream1 = high-res, stream2 = low-res
   TAPO_CAMERA_NAME=office
   TAPO_CAMERA_CAPTURE_DIR=artifacts/camera
   ```

2. Install dependencies (OpenCV already in `requirements.txt`):

   ```
   pip install -r requirements.txt
   ```

   For a headless/server host, `opencv-python-headless` can replace
   `opencv-python`. ONVIF PTZ is optional: `pip install onvif-zeep`.

## Manual test

```
python -m camera --camera office --capture     # capture one frame, print the path
python -m camera --status                       # reachability only, no capture
python -m camera --capture --low-res            # use stream2
```

Credentials are never printed, and the RTSP URL (which contains them) is never
logged.

## Security

- The RTSP URL is built from individual env vars and URL-encoded; it is never
  logged. `camera.redact()` scrubs credentials from any error text.
- Local-network only. No port-forwarding or cloud relay is configured.
- All capture calls have timeouts; the capture resource is released in a `finally`
  block / context manager after every single-frame request.

## Optional ONVIF PTZ (`camera_ptz.py`)

Disabled unless `TAPO_ONVIF_ENABLED=true` and the ONVIF port/credentials are set.
The ONVIF service port is **not** the RTSP port (554) — set `TAPO_ONVIF_PORT`
separately. `OnvifPTZ` exposes `get_ptz_status`, `move_continuous`, `stop`,
`go_to_preset`, and `list_presets`. Not wired into the router; nothing moves the
camera during tests.
