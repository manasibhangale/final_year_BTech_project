from streamlit_mic_recorder import mic_recorder
from modules.audio_loader import transcribe_audio


def voice_input():
    """
    ChatGPT-style voice capture
    Returns transcribed text or None
    """

    audio = mic_recorder(
        start_prompt="🎤 Click & Speak",
        stop_prompt="⏹ Stop & Send",
        key="chatgpt_voice"
    )

    if audio:
        text = transcribe_audio(audio["bytes"])
        return text

    return None