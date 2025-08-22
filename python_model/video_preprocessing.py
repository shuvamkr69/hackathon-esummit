"""
Video preprocessing module for autism detection model
Handles video loading, frame extraction, and feature preprocessing
"""

import cv2
import mediapipe as mp
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import json
import os
from typing import List, Dict, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoPreprocessor:
    """Handles video preprocessing for autism detection model"""
    
    def __init__(self, target_fps: int = 30, target_resolution: Tuple[int, int] = (224, 224)):
        self.target_fps = target_fps
        self.target_resolution = target_resolution
        
        # Initialize MediaPipe
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )
        self.pose_detection = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hands_detection = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        logger.info("VideoPreprocessor initialized successfully")
    
    def extract_frames(self, video_path: str, max_frames: int = 300) -> np.ndarray:
        """
        Extract frames from video with uniform sampling
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to extract
            
        Returns:
            Array of extracted frames
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"Video: {total_frames} frames, {fps} fps")
        
        # Calculate frame sampling interval
        if total_frames > max_frames:
            interval = total_frames // max_frames
        else:
            interval = 1
            
        frames = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % interval == 0:
                # Resize frame
                frame = cv2.resize(frame, self.target_resolution)
                frames.append(frame)
                
            frame_count += 1
            
        cap.release()
        
        if len(frames) == 0:
            raise ValueError("No frames extracted from video")
        
        logger.info(f"Extracted {len(frames)} frames from video")
        return np.array(frames)
    
    def extract_facial_features(self, frame: np.ndarray) -> Dict:
        """
        Extract facial landmarks and features
        
        Args:
            frame: Input frame
            
        Returns:
            Dictionary containing facial features
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        
        features = {
            'face_detected': False,
            'bounding_box': None,
            'confidence': 0.0,
            'num_faces': 0
        }
        
        if results.detections:
            features['face_detected'] = True
            features['num_faces'] = len(results.detections)
            
            # Get the most confident detection
            best_detection = max(results.detections, 
                               key=lambda x: x.score[0])
            
            bbox = best_detection.location_data.relative_bounding_box
            features['bounding_box'] = [bbox.xmin, bbox.ymin, bbox.width, bbox.height]
            features['confidence'] = best_detection.score[0]
            
        return features
    
    def extract_pose_features(self, frame: np.ndarray) -> Dict:
        """
        Extract body pose landmarks
        
        Args:
            frame: Input frame
            
        Returns:
            Dictionary containing pose features
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose_detection.process(rgb_frame)
        
        features = {
            'pose_detected': False,
            'landmarks': None,
            'visibility_scores': None
        }
        
        if results.pose_landmarks:
            features['pose_detected'] = True
            landmarks = []
            visibility_scores = []
            
            for landmark in results.pose_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y, landmark.z])
                visibility_scores.append(landmark.visibility)
                
            features['landmarks'] = np.array(landmarks)
            features['visibility_scores'] = np.array(visibility_scores)
            
        return features
    
    def extract_hand_features(self, frame: np.ndarray) -> Dict:
        """
        Extract hand landmarks for repetitive behavior detection
        
        Args:
            frame: Input frame
            
        Returns:
            Dictionary containing hand features
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands_detection.process(rgb_frame)
        
        features = {
            'hands_detected': False,
            'num_hands': 0,
            'hand_landmarks': [],
            'handedness': []
        }
        
        if results.multi_hand_landmarks:
            features['hands_detected'] = True
            features['num_hands'] = len(results.multi_hand_landmarks)
            
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.append([landmark.x, landmark.y, landmark.z])
                features['hand_landmarks'].append(np.array(landmarks))
            
            if results.multi_handedness:
                for handedness in results.multi_handedness:
                    features['handedness'].append(handedness.classification[0].label)
        
        return features
    
    def process_video(self, video_path: str, max_frames: int = 300) -> Dict:
        """
        Complete video processing pipeline
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to process
            
        Returns:
            Dictionary containing all extracted features
        """
        # Extract frames
        frames = self.extract_frames(video_path, max_frames)
        
        # Initialize feature storage
        facial_features = []
        pose_features = []
        hand_features = []
        
        # Process each frame
        for i, frame in enumerate(frames):
            # Extract features
            face_feat = self.extract_facial_features(frame)
            pose_feat = self.extract_pose_features(frame)
            hand_feat = self.extract_hand_features(frame)
            
            facial_features.append(face_feat)
            pose_features.append(pose_feat)
            hand_features.append(hand_feat)
            
            if (i + 1) % 50 == 0:
                logger.info(f"Processed {i + 1}/{len(frames)} frames")
        
        return {
            'frames': frames,
            'facial_features': facial_features,
            'pose_features': pose_features,
            'hand_features': hand_features,
            'metadata': {
                'num_frames': len(frames),
                'video_path': video_path,
                'target_resolution': self.target_resolution
            }
        }
    
    def cleanup(self):
        """Clean up MediaPipe resources"""
        self.face_detection.close()
        self.pose_detection.close()
        self.hands_detection.close()


class AutismVideoDataset(Dataset):
    """PyTorch dataset for autism detection video data"""
    
    def __init__(self, 
                 data_dir: str, 
                 annotations_file: str,
                 max_frames: int = 300,
                 transform: Optional[transforms.Compose] = None):
        """
        Initialize dataset
        
        Args:
            data_dir: Directory containing video files
            annotations_file: Path to annotations JSON file
            max_frames: Maximum frames per video
            transform: Optional transforms to apply
        """
        self.data_dir = data_dir
        self.max_frames = max_frames
        self.transform = transform
        
        # Load annotations
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        
        self.video_paths = list(self.annotations.keys())
        self.preprocessor = VideoPreprocessor()
        
        logger.info(f"Dataset initialized with {len(self.video_paths)} videos")
    
    def __len__(self) -> int:
        return len(self.video_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Get item from dataset
        
        Args:
            idx: Index of item to retrieve
            
        Returns:
            Tuple of (visual_features, pose_features, labels)
        """
        video_id = self.video_paths[idx]
        video_path = os.path.join(self.data_dir, video_id)
        
        # Process video
        processed_data = self.preprocessor.process_video(video_path, self.max_frames)
        
        # Extract visual features (frames)
        frames = processed_data['frames']
        visual_features = self._frames_to_tensor(frames)
        
        # Extract pose features
        pose_features = self._extract_pose_tensor(processed_data['pose_features'])
        
        # Get labels
        labels = self._extract_labels(video_id)
        
        # Apply transforms if specified
        if self.transform:
            visual_features = self.transform(visual_features)
        
        return visual_features, pose_features, labels
    
    def _frames_to_tensor(self, frames: np.ndarray) -> torch.Tensor:
        """Convert frames to tensor format"""
        # Normalize frames to [0, 1]
        frames = frames.astype(np.float32) / 255.0
        
        # Convert to tensor (T, H, W, C)
        frames_tensor = torch.from_numpy(frames)
        
        # Rearrange to (T, C, H, W)
        frames_tensor = frames_tensor.permute(0, 3, 1, 2)
        
        return frames_tensor
    
    def _extract_pose_tensor(self, pose_features: List[Dict]) -> torch.Tensor:
        """Extract pose landmarks as tensor"""
        pose_data = []
        
        for frame_pose in pose_features:
            if frame_pose['pose_detected'] and frame_pose['landmarks'] is not None:
                # Flatten landmarks (33 landmarks * 3 coordinates = 99 features)
                landmarks = frame_pose['landmarks'].flatten()
                pose_data.append(landmarks)
            else:
                # Use zeros for missing pose data
                pose_data.append(np.zeros(99))
        
        return torch.from_numpy(np.array(pose_data)).float()
    
    def _extract_labels(self, video_id: str) -> Dict[str, torch.Tensor]:
        """Extract labels for video"""
        annotation = self.annotations[video_id]
        
        # Main label (autism indicators present)
        main_label = 1 if annotation['labels']['autism_indicators'] > 0.5 else 0
        
        # Behavioral indicator labels
        behavioral = annotation['annotations']['behavioral_indicators']
        
        eye_contact = 1 if behavioral['eye_contact']['frequency'] == 'low' else 0
        social_engagement = 1 if behavioral['social_interaction']['engagement_level'] == 'minimal' else 0
        repetitive_behavior = 1 if 'repetitive_behaviors' in behavioral else 0
        facial_expression = 1 if behavioral.get('facial_expressions', {}).get('atypical_patterns', False) else 0
        
        return {
            'main_label': torch.tensor(main_label, dtype=torch.long),
            'eye_contact': torch.tensor(eye_contact, dtype=torch.float),
            'social_engagement': torch.tensor(social_engagement, dtype=torch.float),
            'repetitive_behavior': torch.tensor(repetitive_behavior, dtype=torch.float),
            'facial_expression': torch.tensor(facial_expression, dtype=torch.float)
        }


# Data augmentation transforms
def get_video_transforms():
    """Get video data augmentation transforms"""
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


if __name__ == "__main__":
    # Example usage
    preprocessor = VideoPreprocessor()
    
    # Process a sample video
    sample_video = "sample_video.mp4"
    if os.path.exists(sample_video):
        processed_data = preprocessor.process_video(sample_video)
        print(f"Processed video with {processed_data['metadata']['num_frames']} frames")
    else:
        print("Sample video not found")
    
    preprocessor.cleanup()
