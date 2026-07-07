"""speak() via pyttsx3, module-level engine (init once, not per call)."""

import pyttsx3

_engine = pyttsx3.init()
_engine.setProperty("rate", 185)


def speak(text):
    if not text:
        return
    _engine.say(text)
    _engine.runAndWait()


# Future: def speak_piper(text): ...  (Piper/Kokoro TTS is out of scope for v1 — stub only)
