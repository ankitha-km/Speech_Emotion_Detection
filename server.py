import os
import subprocess
import tempfile
import time

import imageio_ffmpeg
from fastapi.responses import FileResponse
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Bilingual Speech Emotion Recognition")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def convert_to_wav(input_path: str) -> str:
    """Convert any uploaded audio to mono 22050 Hz wav (same as app.py)."""
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name

    command = [ffmpeg, "-y", "-i", input_path, "-ar", "22050", "-ac", "1", output_path]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode != 0:
        error = result.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"FFmpeg could not convert the audio: {error[-300:]}")

    return output_path

@app.get("/")
def home():
    return FileResponse("index.html")

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...), language: str = Form("Kannada")):
    if language not in ("Kannada", "English"):
        raise HTTPException(status_code=400, detail="Language must be Kannada or English.")

    input_path = None
    wav_path = None

    try:
        extension = os.path.splitext(file.filename or "")[1] or ".webm"
        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp:
            tmp.write(await file.read())
            input_path = tmp.name

        wav_path = convert_to_wav(input_path)

        start = time.time()

        if language == "English":
            # Loaded on first use, so only the English model is needed here
            from ser_english import predict_english

            emotion, prob_dict = predict_english(wav_path)
            probs = {str(k): float(v) for k, v in prob_dict.items()}
        else:
            # Loaded on first use, so only the Kannada model is needed here
            from inference import predict_emotion, CLASSES

            emotion, probabilities = predict_emotion(wav_path)
            probs = {str(c): float(p) for c, p in zip(CLASSES, probabilities)}

        latency = round(time.time() - start, 2)

        return {
            "emotion": str(emotion),
            "probs": probs,
            "latency": latency,
            "language": language,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        for p in (input_path, wav_path):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass