import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
import json

class AutismDetector:
    def __init__(self, model_path, meta_path, device=None):
        # Load metadata
        with open(meta_path, "r") as f:
            self.meta = json.load(f)
        self.classes = self.meta["classes"]
        self.input_size = self.meta.get("input_size", 224)

        # Device setup
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Recreate architecture EXACTLY as in training
        self.model = timm.create_model(
            "rexnet_150", pretrained=False, num_classes=len(self.classes)
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # Preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def predict(self, frame):
        # Convert OpenCV BGR → PIL RGB
        img = Image.fromarray(frame[..., ::-1])
        img_t = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(img_t)
            probs = torch.softmax(outputs, dim=1)[0]
            pred_idx = torch.argmax(probs).item()

        return {
            "class": self.classes[pred_idx],
            "confidence": float(probs[pred_idx].item())
        }
