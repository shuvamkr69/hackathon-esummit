"""
Feature extraction module for autism detection model
Handles visual feature extraction using pre-trained CNNs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
import timm
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class VisualFeatureExtractor:
    """Extract visual features from video frames using pre-trained CNNs"""
    
    def __init__(self, 
                 model_name: str = 'resnet50',
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize feature extractor
        
        Args:
            model_name: Name of pre-trained model to use
            device: Device to run model on
        """
        self.device = device
        self.model_name = model_name
        
        # Load pre-trained model
        if model_name == 'resnet50':
            self.model = models.resnet50(pretrained=True)
            self.model.fc = nn.Identity()  # Remove final classification layer
            self.feature_dim = 2048
        elif model_name == 'efficientnet_b0':
            self.model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
            self.feature_dim = 1280
        elif model_name == 'vit_base_patch16_224':
            self.model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
            self.feature_dim = 768
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        self.model.to(device)
        self.model.eval()
        
        # Define preprocessing transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        logger.info(f"VisualFeatureExtractor initialized with {model_name}")
    
    def extract_frame_features(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract features from a single frame
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            Feature vector as numpy array
        """
        # Preprocess frame
        frame_tensor = self.transform(frame).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model(frame_tensor)
            
        return features.cpu().numpy().squeeze()
    
    def extract_video_features(self, frames: np.ndarray) -> np.ndarray:
        """
        Extract features from all frames in a video
        
        Args:
            frames: Array of frames (T, H, W, C)
            
        Returns:
            Feature matrix (T, feature_dim)
        """
        features = []
        
        for i, frame in enumerate(frames):
            frame_features = self.extract_frame_features(frame)
            features.append(frame_features)
            
            if (i + 1) % 50 == 0:
                logger.info(f"Extracted features from {i + 1}/{len(frames)} frames")
        
        return np.array(features)
    
    def extract_temporal_features(self, features: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract temporal patterns from frame features
        
        Args:
            features: Frame features array (T, feature_dim)
            
        Returns:
            Dictionary containing temporal features
        """
        # Calculate temporal derivatives
        temporal_diff = np.diff(features, axis=0)
        
        # Calculate moving averages
        window_size = min(10, len(features) // 4)
        if window_size > 1:
            moving_avg = np.array([
                np.convolve(features[:, i], 
                           np.ones(window_size) / window_size, 
                           mode='valid')
                for i in range(features.shape[1])
            ]).T
        else:
            moving_avg = features
        
        # Calculate variance over time
        temporal_variance = np.var(features, axis=0)
        
        # Calculate mean and std
        temporal_mean = np.mean(features, axis=0)
        temporal_std = np.std(features, axis=0)
        
        return {
            'temporal_diff': temporal_diff,
            'moving_average': moving_avg,
            'variance': temporal_variance,
            'mean': temporal_mean,
            'std': temporal_std
        }


class BehavioralFeatureExtractor:
    """Extract behavioral features from pose and hand landmarks"""
    
    def __init__(self):
        """Initialize behavioral feature extractor"""
        # Define important pose landmarks indices
        self.face_landmarks = list(range(0, 11))  # Face landmarks
        self.upper_body_landmarks = list(range(11, 17))  # Shoulders, elbows, wrists
        self.hand_landmarks = [15, 16, 17, 18, 19, 20, 21, 22]  # Hand landmarks
        
        logger.info("BehavioralFeatureExtractor initialized")
    
    def extract_eye_contact_features(self, facial_features: List[Dict]) -> Dict[str, float]:
        """
        Extract eye contact related features
        
        Args:
            facial_features: List of facial features per frame
            
        Returns:
            Dictionary containing eye contact metrics
        """
        face_detection_ratio = sum(1 for f in facial_features if f['face_detected']) / len(facial_features)
        
        # Calculate average confidence when face is detected
        confidences = [f['confidence'] for f in facial_features if f['face_detected']]
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        # Calculate face size consistency (proxy for looking at camera)
        face_sizes = []
        for f in facial_features:
            if f['face_detected'] and f['bounding_box']:
                bbox = f['bounding_box']
                size = bbox[2] * bbox[3]  # width * height
                face_sizes.append(size)
        
        face_size_variance = np.var(face_sizes) if face_sizes else 0.0
        
        return {
            'face_detection_ratio': face_detection_ratio,
            'avg_face_confidence': avg_confidence,
            'face_size_variance': face_size_variance,
            'consistent_face_presence': face_detection_ratio > 0.7
        }
    
    def extract_movement_features(self, pose_features: List[Dict]) -> Dict[str, float]:
        """
        Extract movement and repetitive behavior features
        
        Args:
            pose_features: List of pose features per frame
            
        Returns:
            Dictionary containing movement metrics
        """
        # Extract valid pose sequences
        valid_poses = [p for p in pose_features if p['pose_detected']]
        
        if len(valid_poses) < 2:
            return {
                'movement_variance': 0.0,
                'repetitive_motion_score': 0.0,
                'upper_body_activity': 0.0,
                'hand_movement_frequency': 0.0
            }
        
        # Calculate movement variance
        landmarks_sequence = np.array([p['landmarks'] for p in valid_poses])
        movement_diffs = np.diff(landmarks_sequence, axis=0)
        movement_variance = np.var(movement_diffs)
        
        # Calculate repetitive motion score
        # Look for periodic patterns in movement
        upper_body_sequence = landmarks_sequence[:, self.upper_body_landmarks, :]
        upper_body_movement = np.diff(upper_body_sequence, axis=0)
        upper_body_variance = np.var(upper_body_movement, axis=0)
        
        # Calculate hand movement frequency
        if len(self.hand_landmarks) > 0:
            hand_sequence = landmarks_sequence[:, [idx for idx in self.hand_landmarks if idx < landmarks_sequence.shape[1]], :]
            hand_movement = np.diff(hand_sequence, axis=0)
            hand_movement_freq = np.sum(np.abs(hand_movement) > 0.05) / len(hand_movement)
        else:
            hand_movement_freq = 0.0
        
        # Detect repetitive patterns using autocorrelation
        def detect_repetitive_patterns(signal):
            if len(signal) < 4:
                return 0.0
            # Simple autocorrelation-based repetition detection
            autocorr = np.correlate(signal, signal, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            # Look for peaks indicating repetition
            peaks = []
            for i in range(1, min(len(autocorr)-1, len(signal)//2)):
                if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                    peaks.append(autocorr[i])
            return np.max(peaks) if peaks else 0.0
        
        repetitive_score = 0.0
        if len(upper_body_movement) > 0:
            # Calculate repetitive score for each joint
            for joint_idx in range(upper_body_movement.shape[1]):
                for coord_idx in range(upper_body_movement.shape[2]):
                    signal = upper_body_movement[:, joint_idx, coord_idx]
                    repetitive_score += detect_repetitive_patterns(signal)
        
        return {
            'movement_variance': float(movement_variance),
            'repetitive_motion_score': float(repetitive_score),
            'upper_body_activity': float(np.mean(upper_body_variance)),
            'hand_movement_frequency': float(hand_movement_freq)
        }
    
    def extract_social_engagement_features(self, 
                                         facial_features: List[Dict],
                                         pose_features: List[Dict]) -> Dict[str, float]:
        """
        Extract social engagement related features
        
        Args:
            facial_features: List of facial features per frame
            pose_features: List of pose features per frame
            
        Returns:
            Dictionary containing social engagement metrics
        """
        # Calculate face orientation consistency
        face_orientations = []
        for f in facial_features:
            if f['face_detected'] and f['bounding_box']:
                bbox = f['bounding_box']
                # Center position of face
                center_x = bbox[0] + bbox[2] / 2
                center_y = bbox[1] + bbox[3] / 2
                face_orientations.append([center_x, center_y])
        
        orientation_variance = np.var(face_orientations, axis=0) if face_orientations else [1.0, 1.0]
        
        # Calculate body orientation towards camera/audience
        body_orientations = []
        for p in pose_features:
            if p['pose_detected'] and p['landmarks'] is not None:
                landmarks = p['landmarks']
                if len(landmarks) > 12:  # Ensure we have shoulder landmarks
                    left_shoulder = landmarks[11]
                    right_shoulder = landmarks[12]
                    shoulder_angle = np.arctan2(
                        right_shoulder[1] - left_shoulder[1],
                        right_shoulder[0] - left_shoulder[0]
                    )
                    body_orientations.append(shoulder_angle)
        
        body_orientation_variance = np.var(body_orientations) if body_orientations else 1.0
        
        return {
            'face_orientation_consistency': 1.0 / (1.0 + np.mean(orientation_variance)),
            'body_orientation_consistency': 1.0 / (1.0 + body_orientation_variance),
            'engagement_score': len(face_orientations) / len(facial_features) if facial_features else 0.0
        }


class ComprehensiveFeatureExtractor:
    """Combines all feature extraction modules"""
    
    def __init__(self, 
                 visual_model: str = 'resnet50',
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize comprehensive feature extractor
        
        Args:
            visual_model: Name of visual model to use
            device: Device to run models on
        """
        self.visual_extractor = VisualFeatureExtractor(visual_model, device)
        self.behavioral_extractor = BehavioralFeatureExtractor()
        self.device = device
        
        logger.info("ComprehensiveFeatureExtractor initialized")
    
    def extract_all_features(self, processed_video_data: Dict) -> Dict:
        """
        Extract all features from processed video data
        
        Args:
            processed_video_data: Output from VideoPreprocessor.process_video()
            
        Returns:
            Dictionary containing all extracted features
        """
        frames = processed_video_data['frames']
        facial_features = processed_video_data['facial_features']
        pose_features = processed_video_data['pose_features']
        hand_features = processed_video_data['hand_features']
        
        # Extract visual features
        logger.info("Extracting visual features...")
        visual_features = self.visual_extractor.extract_video_features(frames)
        temporal_features = self.visual_extractor.extract_temporal_features(visual_features)
        
        # Extract behavioral features
        logger.info("Extracting behavioral features...")
        eye_contact_features = self.behavioral_extractor.extract_eye_contact_features(facial_features)
        movement_features = self.behavioral_extractor.extract_movement_features(pose_features)
        social_features = self.behavioral_extractor.extract_social_engagement_features(
            facial_features, pose_features
        )
        
        # Combine all features
        all_features = {
            'visual_features': visual_features,
            'temporal_features': temporal_features,
            'eye_contact_features': eye_contact_features,
            'movement_features': movement_features,
            'social_engagement_features': social_features,
            'metadata': {
                'visual_feature_dim': visual_features.shape[1],
                'num_frames': len(frames),
                'model_used': self.visual_extractor.model_name
            }
        }
        
        logger.info("Feature extraction completed")
        return all_features


if __name__ == "__main__":
    # Example usage
    feature_extractor = ComprehensiveFeatureExtractor()
    
    # Create sample data
    sample_frames = np.random.randint(0, 255, (10, 224, 224, 3), dtype=np.uint8)
    sample_facial = [{'face_detected': True, 'confidence': 0.8, 'bounding_box': [0.1, 0.1, 0.3, 0.4]} for _ in range(10)]
    sample_pose = [{'pose_detected': True, 'landmarks': np.random.rand(33, 3)} for _ in range(10)]
    
    sample_data = {
        'frames': sample_frames,
        'facial_features': sample_facial,
        'pose_features': sample_pose,
        'hand_features': []
    }
    
    # Extract features
    features = feature_extractor.extract_all_features(sample_data)
    print(f"Extracted features with visual shape: {features['visual_features'].shape}")
    print(f"Eye contact features: {features['eye_contact_features']}")
    print(f"Movement features: {features['movement_features']}")
    print(f"Social engagement features: {features['social_engagement_features']}")
