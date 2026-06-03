from faster_whisper import WhisperModel
import tempfile
import soundfile as sf

# Load model once (small = fast)
model = WhisperModel("base", compute_type="int8")


def transcribe_audio(audio_file):
    """
    Converts voice → text
    """

    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_file.read())
            tmp_path = tmp.name

        segments, _ = model.transcribe(tmp_path)

        text = " ".join([seg.text for seg in segments])

        return text

    except Exception as e:
        return f"Error in voice input: {str(e)}"