from facenet_pytorch import MTCNN
import cv2

mtcnn = MTCNN(keep_all=True)

def analyze_gaze(frame):
    """Very basic gaze estimation using eye position symmetry"""
    boxes, _ = mtcnn.detect(frame)
    if boxes is None:
        return "No face detected"

    gaze_results = []
    for box in boxes:
        x1, y1, x2, y2 = [int(b) for b in box]
        face = frame[y1:y2, x1:x2]

        # For demo: assume gaze is "forward" if face width/height ratio > 0.8
        ratio = (x2-x1) / (y2-y1)
        if ratio > 0.8:
            gaze_results.append("Forward")
        else:
            gaze_results.append("Not Forward")

    return gaze_results
