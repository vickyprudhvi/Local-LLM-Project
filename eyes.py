"""snapshot, describe_local (moondream), describe_claude."""

import base64
import os

import cv2
import requests
from rich.console import Console

import brain

console = Console()

OLLAMA_URL = "http://localhost:11434"
VISION_MODEL = os.environ.get("VISION_MODEL", "moondream")


def snapshot(path="snapshots/latest.jpg", camera_index=0):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # CAP_DSHOW returns unusable near-black frames on this machine's webcam driver;
    # CAP_MSMF is the working backend here.
    cap = cv2.VideoCapture(camera_index, cv2.CAP_MSMF)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera index {camera_index}.")

        # discard the first several frames — many webcams return black frames
        # while auto-exposure ramps up right after the camera opens
        frame = None
        for _ in range(15):
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"Could not read a frame from camera index {camera_index}.")
    finally:
        cap.release()

    cv2.imwrite(path, frame)
    return path


def _resize_and_encode(path, max_side, quality=85):
    image = cv2.imread(path)
    if image is None:
        raise RuntimeError(f"Could not read image at {path}.")

    height, width = image.shape[:2]
    longest = max(height, width)
    if longest > max_side:
        scale = max_side / longest
        image = cv2.resize(image, (int(width * scale), int(height * scale)))

    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError(f"Could not encode image at {path}.")

    return base64.b64encode(buffer).decode("utf-8")


def describe_local(path, prompt="Describe what you see."):
    b64 = _resize_and_encode(path, max_side=1280)

    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": VISION_MODEL,
                "prompt": prompt,
                "images": [b64],
                "stream": False,
            },
            timeout=60,
        )
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        console.print(f"[red]Vision model call failed: {e}[/red]")
        return "Sorry, I couldn't look at that just now."

    return resp.json().get("response", "").strip()


def describe_claude(path, prompt, history=None, system_prompt=""):
    # ask_claude reads and encodes the file itself; resize to Claude's larger limit first.
    resized_b64 = _resize_and_encode(path, max_side=1600)
    tmp_path = os.path.join(os.path.dirname(path) or ".", "_claude_resized.jpg")
    with open(tmp_path, "wb") as f:
        f.write(base64.b64decode(resized_b64))

    return brain.ask_claude(prompt, history or [], system_prompt, image_path=tmp_path)
