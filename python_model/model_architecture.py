"""
Model architecture for autism detection
Multi-modal deep learning model combining visual, pose, and behavioral features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class AutismDetectionModel(nn.Module):
    """
    Multi-modal deep learning model for autism detection
    Combines visual features, pose features, and behavioral indicators
    """
    
    def __init__(self, 
                 visual_feature_dim: int = 2048,
                 pose_feature_dim: int = 99,  # 33 landmarks * 3 coordinates
                 temporal_length: int = 300,
                 hidden_dim: int = 512,
                 num_classes: int = 2,
                 dropout_rate: float = 0.3,
                 use_attention: bool = True):
        """
        Initialize the autism detection model
        
        Args:
            visual_feature_dim: Dimension of visual features
            pose_feature_dim: Dimension of pose features
            temporal_length: Maximum sequence length
            hidden_dim: Hidden dimension size
            num_classes: Number of output classes
            dropout_rate: Dropout rate for regularization
            use_attention: Whether to use attention mechanism
        """
        super(AutismDetectionModel, self).__init__()
        
        self.visual_feature_dim = visual_feature_dim
        self.pose_feature_dim = pose_feature_dim
        self.temporal_length = temporal_length
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.use_attention = use_attention
        
        # Visual feature processing
        self.visual_encoder = nn.Sequential(
            nn.Linear(visual_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Pose feature processing
        self.pose_encoder = nn.Sequential(
            nn.Linear(pose_feature_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Combined feature dimension
        combined_dim = hidden_dim // 2 + hidden_dim // 4
        
        # Temporal modeling with LSTM
        self.temporal_lstm = nn.LSTM(
            input_size=combined_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout_rate if dropout_rate > 0 else 0,
            bidirectional=True
        )
        
        lstm_output_dim = hidden_dim * 2  # Bidirectional
        
        # Attention mechanism
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=lstm_output_dim,
                num_heads=8,
                dropout=dropout_rate,
                batch_first=True
            )
            attention_output_dim = lstm_output_dim
        else:
            attention_output_dim = lstm_output_dim
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(attention_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Behavioral indicator heads
        self.eye_contact_head = nn.Sequential(
            nn.Linear(attention_output_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        self.social_engagement_head = nn.Sequential(
            nn.Linear(attention_output_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        self.repetitive_behavior_head = nn.Sequential(
            nn.Linear(attention_output_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        self.facial_expression_head = nn.Sequential(
            nn.Linear(attention_output_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Confidence estimation head
        self.confidence_head = nn.Sequential(
            nn.Linear(attention_output_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        logger.info(f"AutismDetectionModel initialized with {self._count_parameters()} parameters")
    
    def _init_weights(self, module):
        """Initialize model weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
    
    def _count_parameters(self):
        """Count total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, visual_features: torch.Tensor, pose_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model
        
        Args:
            visual_features: Visual features tensor (batch_size, seq_len, visual_dim)
            pose_features: Pose features tensor (batch_size, seq_len, pose_dim)
            
        Returns:
            Dictionary containing all model outputs
        """
        batch_size, seq_len, _ = visual_features.shape
        
        # Encode visual features
        visual_encoded = self.visual_encoder(visual_features)
        
        # Encode pose features
        pose_encoded = self.pose_encoder(pose_features)
        
        # Concatenate features
        combined_features = torch.cat([visual_encoded, pose_encoded], dim=-1)
        
        # Temporal modeling with LSTM
        lstm_out, (h_n, c_n) = self.temporal_lstm(combined_features)
        
        # Apply attention if enabled
        if self.use_attention:
            attended_features, attention_weights = self.attention(
                lstm_out, lstm_out, lstm_out
            )
        else:
            attended_features = lstm_out
            attention_weights = None
        
        # Global average pooling over time dimension
        pooled_features = torch.mean(attended_features, dim=1)
        
        # Generate predictions
        main_output = self.classifier(pooled_features)
        
        # Behavioral indicators
        eye_contact = self.eye_contact_head(pooled_features)
        social_engagement = self.social_engagement_head(pooled_features)
        repetitive_behavior = self.repetitive_behavior_head(pooled_features)
        facial_expression = self.facial_expression_head(pooled_features)
        
        # Confidence estimation
        confidence = self.confidence_head(pooled_features)
        
        outputs = {
            'main_output': main_output,
            'eye_contact': eye_contact,
            'social_engagement': social_engagement,
            'repetitive_behavior': repetitive_behavior,
            'facial_expression': facial_expression,
            'confidence': confidence
        }
        
        if attention_weights is not None:
            outputs['attention_weights'] = attention_weights
        
        return outputs
    
    def get_feature_importance(self, visual_features: torch.Tensor, pose_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get feature importance scores using gradient-based methods
        
        Args:
            visual_features: Visual features tensor
            pose_features: Pose features tensor
            
        Returns:
            Dictionary containing feature importance scores
        """
        self.eval()
        
        # Enable gradients for input features
        visual_features.requires_grad_(True)
        pose_features.requires_grad_(True)
        
        # Forward pass
        outputs = self.forward(visual_features, pose_features)
        
        # Calculate gradients with respect to main output
        main_output = outputs['main_output']
        autism_score = torch.softmax(main_output, dim=1)[:, 1]  # Autism probability
        
        # Backpropagate to get gradients
        grad_outputs = torch.ones_like(autism_score)
        visual_grads, pose_grads = torch.autograd.grad(
            outputs=autism_score,
            inputs=[visual_features, pose_features],
            grad_outputs=grad_outputs,
            create_graph=False,
            retain_graph=False
        )
        
        # Calculate importance scores
        visual_importance = torch.abs(visual_grads).mean(dim=0)  # Average over batch
        pose_importance = torch.abs(pose_grads).mean(dim=0)
        
        return {
            'visual_importance': visual_importance,
            'pose_importance': pose_importance
        }


class EarlyFusionModel(nn.Module):
    """
    Alternative model architecture using early fusion of features
    """
    
    def __init__(self, 
                 visual_feature_dim: int = 2048,
                 pose_feature_dim: int = 99,
                 hidden_dim: int = 512,
                 num_classes: int = 2):
        super(EarlyFusionModel, self).__init__()
        
        # Combine features early
        combined_dim = visual_feature_dim + pose_feature_dim
        
        self.feature_fusion = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Temporal processing
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Classification
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, visual_features: torch.Tensor, pose_features: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = visual_features.shape
        
        # Concatenate features
        combined = torch.cat([visual_features, pose_features], dim=-1)
        
        # Fuse features
        fused = self.feature_fusion(combined)
        
        # Temporal processing (B, T, D) -> (B, D, T)
        fused = fused.transpose(1, 2)
        temporal_out = self.temporal_conv(fused)
        temporal_out = temporal_out.squeeze(-1)
        
        # Classification
        output = self.classifier(temporal_out)
        
        return output


class TransformerModel(nn.Module):
    """
    Transformer-based model for autism detection
    """
    
    def __init__(self,
                 visual_feature_dim: int = 2048,
                 pose_feature_dim: int = 99,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_layers: int = 6,
                 num_classes: int = 2):
        super(TransformerModel, self).__init__()
        
        self.d_model = d_model
        
        # Feature projection
        self.visual_proj = nn.Linear(visual_feature_dim, d_model // 2)
        self.pose_proj = nn.Linear(pose_feature_dim, d_model // 2)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_classes)
        )
    
    def forward(self, visual_features: torch.Tensor, pose_features: torch.Tensor) -> torch.Tensor:
        # Project features
        visual_proj = self.visual_proj(visual_features)
        pose_proj = self.pose_proj(pose_features)
        
        # Combine features
        combined = torch.cat([visual_proj, pose_proj], dim=-1)
        
        # Add positional encoding
        combined = self.pos_encoding(combined)
        
        # Transformer encoding
        encoded = self.transformer(combined)
        
        # Global average pooling
        pooled = torch.mean(encoded, dim=1)
        
        # Classification
        output = self.classifier(pooled)
        
        return output


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer model"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :].transpose(0, 1)
        return x


def create_model(model_type: str = 'multimodal', **kwargs) -> nn.Module:
    """
    Factory function to create different model architectures
    
    Args:
        model_type: Type of model to create
        **kwargs: Additional arguments for model initialization
        
    Returns:
        Initialized model
    """
    if model_type == 'multimodal':
        return AutismDetectionModel(**kwargs)
    elif model_type == 'early_fusion':
        return EarlyFusionModel(**kwargs)
    elif model_type == 'transformer':
        return TransformerModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test model creation and forward pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = AutismDetectionModel().to(device)
    
    # Create sample inputs
    batch_size, seq_len = 4, 100
    visual_features = torch.randn(batch_size, seq_len, 2048).to(device)
    pose_features = torch.randn(batch_size, seq_len, 99).to(device)
    
    # Forward pass
    outputs = model(visual_features, pose_features)
    
    print("Model outputs:")
    for key, value in outputs.items():
        if key != 'attention_weights':
            print(f"{key}: {value.shape}")
    
    # Test feature importance
    importance = model.get_feature_importance(visual_features[:1], pose_features[:1])
    print(f"Visual importance shape: {importance['visual_importance'].shape}")
    print(f"Pose importance shape: {importance['pose_importance'].shape}")
    
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
