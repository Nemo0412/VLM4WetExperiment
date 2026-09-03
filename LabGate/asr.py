"""Audio → text. FineBio eval usually passes ASR strings; wav uses Whisper if installed."""

from __future__ import annotations

from pathlib import Path


def audio_to_text(asr_text: str | None = None, audio_path: str | None = None) -> str:
    if asr_text is not None:
        return str(asr_text).strip()
    if not audio_path:
        return ""
    path = Path(audio_path)
    if not path.exists():
        return ""
    try:
        import whisper  # type: ignore
    except ImportError:
        return ""
    model = whisper.load_model("tiny")
    out = model.transcribe(str(path))
    return str(out.get("text") or "").strip()
