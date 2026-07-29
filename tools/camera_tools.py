"""Router-dispatched built-ins: the camera capabilities.

  camera.look           — laptop webcam snapshot, described by the local vision model
  camera.look_carefully — laptop webcam snapshot, read carefully by Claude vision
  camera.capture        — single still from the networked Tapo camera at its current aim
  camera.scan           — pan/tilt sweep of the room, then answer from the views

Each tool performs only the bounded capture/hardware step and returns a render
directive naming how the orchestrator should turn the image(s) into a spoken
answer (describe_local / describe_claude / scan_synthesize). The vision- and
summarization-LLM calls themselves live in tool_dispatch, so these tools stay
free of conversation history, system prompts, and reply wording. Camera/scan
faults are caught here and returned as a plain speakable message, preserving the
exact wording of the former assistant.dispatch branches. llm_callable=False:
router-selected only, never offered to the local tool-calling loop.
"""

import camera
import camera_ptz
import eyes
from tools.base import BaseTool
from tools.models import ToolPermission

# Raised above eyes.describe_local's normal 1280px default: a stitched room
# panorama benefits from finer detail, and since scan sends one merged image
# instead of several, the extra resolution doesn't multiply per-frame cost.
SCAN_ROOM_VISION_MAX_SIDE = 1920


class LookTool(BaseTool):
    name = "camera.look"
    description = "Take a photo with the laptop webcam and describe what is in it."
    input_schema = {"type": "object", "properties": {}}
    timeout_seconds = 60.0
    llm_callable = False
    # READ: overwrites a single fixed transient buffer (snapshots/latest.jpg); it
    # does not move hardware or accumulate persistent files. (Ambiguous — if this
    # were changed to save timestamped/persistent captures, reclassify as WRITE.)
    permission = ToolPermission.READ

    def execute(self, arguments: dict) -> dict:
        try:
            path = eyes.snapshot()
        except RuntimeError as e:
            return {"render": "speak", "text": f"Sorry, I couldn't use the camera: {e}"}
        return {"render": "describe_local", "image_path": path}


class LookCarefullyTool(BaseTool):
    name = "camera.look_carefully"
    description = "Take a photo with the laptop webcam and read it carefully with Claude vision."
    input_schema = {"type": "object", "properties": {}}
    timeout_seconds = 60.0
    llm_callable = False
    # READ: same transient fixed-path snapshot as camera.look (no movement, no
    # persistent files). The image is sent to Claude for description only.
    permission = ToolPermission.READ

    def execute(self, arguments: dict) -> dict:
        try:
            path = eyes.snapshot()
        except RuntimeError as e:
            return {"render": "speak", "text": f"Sorry, I couldn't use the camera: {e}"}
        return {"render": "describe_claude", "image_path": path}


class CaptureCameraTool(BaseTool):
    name = "camera.capture"
    description = "Capture a single still frame from the networked room camera at its current aim."
    input_schema = {
        "type": "object",
        "properties": {"camera_name": {"type": "string", "description": "Camera name, e.g. 'office'."}},
    }
    timeout_seconds = 60.0
    llm_callable = False
    # WRITE: persists a timestamped JPEG to the capture folder (accumulating files
    # on disk = recorded media), so it requires confirmation.
    permission = ToolPermission.WRITE

    def confirmation_summary(self, arguments: dict) -> str:
        name = arguments.get("camera_name") or "office"
        return f"Capture a still frame from the {name} camera and save it to the capture folder."

    def execute(self, arguments: dict) -> dict:
        camera_name = arguments.get("camera_name")
        result = camera.capture_camera_frame(camera_name=camera_name or "office")
        if not result["success"]:
            return {
                "render": "speak",
                "text": f"Sorry, I couldn't capture the {result['camera_name']} camera: {result['error']}",
            }
        # Vision integration point: the captured frame is handed to the local
        # vision processor by the orchestrator (describe_local).
        return {"render": "describe_local", "image_path": result["image_path"]}


class ScanRoomTool(BaseTool):
    name = "camera.scan"
    description = "Pan and tilt the room camera across several views, then answer from them."
    input_schema = {"type": "object", "properties": {}}
    # Physically sweeps the camera with settle delays and stitching — generous cap.
    timeout_seconds = 180.0
    llm_callable = False
    # WRITE: physically moves the PTZ camera (and saves captured frames), so it
    # requires confirmation.
    permission = ToolPermission.WRITE

    def confirmation_summary(self, arguments: dict) -> str:
        return "Physically pan and tilt the room camera to sweep the room, then describe what it sees."

    def execute(self, arguments: dict) -> dict:
        scan = camera_ptz.scan_room()
        if not scan["success"]:
            return {"render": "speak", "text": f"Sorry, I couldn't scan the room: {scan['error']}"}

        if scan["panorama_path"]:
            # Stitched into one image — a single vision call answers directly.
            return {
                "render": "describe_local",
                "image_path": scan["panorama_path"],
                "max_side": SCAN_ROOM_VISION_MAX_SIDE,
            }

        # Stitching failed — describe each frame separately and synthesize one
        # summary (the per-image-then-synthesize pattern plant_watcher.py uses).
        return {
            "render": "scan_synthesize",
            "images": [
                {"position": image["position"], "image_path": image["image_path"]}
                for image in scan["images"]
            ],
        }
