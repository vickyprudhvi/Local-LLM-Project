"""Record + transcribe. Push-to-talk is the default; VAD is experimental."""

import os

import numpy as np
import sounddevice as sd
import soundfile as sf
import webrtcvad
from faster_whisper import WhisperModel
from rich.console import Console

console = Console()

SAMPLE_RATE = 16000
WHISPER_MODEL_NAME = os.environ.get("WHISPER_MODEL", "base.en")

_whisper_model = None


def _get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = WhisperModel(WHISPER_MODEL_NAME, device="cpu", compute_type="int8")
    return _whisper_model


def _transcribe(wav_path):
    model = _get_whisper_model()
    segments, _info = model.transcribe(wav_path, language="en")
    segments = list(segments)  # force-evaluate the generator before joining
    return " ".join(seg.text.strip() for seg in segments).strip()


def listen_push_to_talk(seconds=6):
    input(f"Press Enter, then speak (recording for {seconds}s)...")
    console.print("[dim]recording...[/dim]")
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="int16")
    sd.wait()
    console.print("[dim]transcribing...[/dim]")

    wav_path = "ptt_capture.wav"
    sf.write(wav_path, audio, SAMPLE_RATE, subtype="PCM_16")
    return _transcribe(wav_path)


def listen_vad_webrtc():
    """Experimental. 30ms frames, stop after 900ms of trailing silence, 15s max."""
    frame_ms = 30
    frame_samples = int(SAMPLE_RATE * frame_ms / 1000)
    silence_limit_ms = 900
    max_ms = 15000

    vad = webrtcvad.Vad(2)

    console.print("[dim]listening (VAD, experimental)...[/dim]")
    frames = []
    speech_started = False
    silence_ms = 0
    elapsed_ms = 0

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=frame_samples) as stream:
        while elapsed_ms < max_ms:
            block, _overflowed = stream.read(frame_samples)
            raw = block.tobytes()
            is_speech = vad.is_speech(raw, SAMPLE_RATE)
            frames.append(raw)
            elapsed_ms += frame_ms

            if is_speech:
                speech_started = True
                silence_ms = 0
            elif speech_started:
                silence_ms += frame_ms
                if silence_ms >= silence_limit_ms:
                    break

    raw_audio = b"".join(frames)
    audio = np.frombuffer(raw_audio, dtype="int16")

    wav_path = "vad_capture.wav"
    sf.write(wav_path, audio, SAMPLE_RATE, subtype="PCM_16")
    return _transcribe(wav_path)
