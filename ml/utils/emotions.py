from deepface import DeepFace

def analyze_emotions(frame):
    try:
        analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        return analysis[0]['dominant_emotion']
    except Exception:
        return "Unknown"
