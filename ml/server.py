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
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
autism_detector = AutismDetector(model_path="artifacts/autism_model.pth", meta_path="artifacts/meta.json")

@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await file.read())
        video_path = tmp.name
    frames = extract_frames(video_path, frame_skip=15) 

    results = {
        "autism_predictions": [],
        "gaze": [],
        "emotions": []
    }

    for frame in frames:
        pred = autism_detector.predict(frame)
        results["autism_predictions"].append(pred)

        gaze = analyze_gaze(frame)
        results["gaze"].append(gaze)

        emotion = analyze_emotions(frame)
        results["emotions"].append(emotion)

    os.remove(video_path)  
    return results


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
