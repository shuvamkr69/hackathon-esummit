# FastAPI server for autism detection video analysis
from fastapi import FastAPI, UploadFile, File
import uvicorn
import os
import tempfile

from utils.video import extract_frames
from utils.autism import AutismDetector
from utils.gaze import analyze_gaze
from utils.emotions import analyze_emotions

app = FastAPI(title="Autism Detection Service")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the autism detection model once on startup
autism_detector = AutismDetector(model_path="artifacts/autism_model.pth", meta_path="artifacts/meta.json")

@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    # Save uploaded video temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await file.read())
        video_path = tmp.name

    # Step 1: Extract frames
    frames = extract_frames(video_path, frame_skip=15)  # sample every 15th frame

    results = {
        "autism_predictions": [],
        "gaze": [],
        "emotions": []
    }

    # Step 2: Run analysis on frames
    for frame in frames:
        # Autism prediction
        pred = autism_detector.predict(frame)
        results["autism_predictions"].append(pred)

        # Gaze analysis
        gaze = analyze_gaze(frame)
        results["gaze"].append(gaze)

        # Emotion analysis
        emotion = analyze_emotions(frame)
        results["emotions"].append(emotion)

    os.remove(video_path)  # cleanup temp file
    return results


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
