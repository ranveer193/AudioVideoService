import os, tempfile, time, pathlib
from fastapi import FastAPI, UploadFile, File, HTTPException
from faster_whisper import WhisperModel

MODEL_NAME = os.getenv("WHISPER_MODEL", "Systran/faster-whisper-small.en")

print(f"🔄 Loading {MODEL_NAME} …")
model = WhisperModel(
    MODEL_NAME,
    device="cpu",          # Render free tier = CPU only
    compute_type="int8"    # fastest + <1 GB RAM
)
print("✅ Model ready")

app = FastAPI(title="Whisper-Small EN API")

@app.get("/")
def root():
    return {"status": "ok", "model": MODEL_NAME}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(400, "No file")
    suffix = pathlib.Path(file.filename).suffix or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read()); tmp_path = tmp.name

    t0 = time.time()
    segments, _ = model.transcribe(
        tmp_path,
        language="en",
        beam_size=3,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 400}
    )
    os.remove(tmp_path)

    transcript = " ".join(seg.text.strip() for seg in segments)
    print(f"📝 Transcribed in {time.time()-t0:.1f}s | chars={len(transcript)}")
    return {"transcript": transcript}
