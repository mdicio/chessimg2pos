import torch.nn as nn
from .constants import DEFAULT_USE_GRAYSCALE

class ChessPieceClassifier(nn.Module):
    """Chess piece classifier model"""

    def __init__(self, num_classes=13, use_grayscale=DEFAULT_USE_GRAYSCALE):
        super(ChessPieceClassifier, self).__init__()
        input_channels = 1 if use_grayscale else 3

        self.conv_layers = nn.Sequential(
            # First convolutional block
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Second convolutional block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Third convolutional block
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 64),  # After 2 max pooling, 32x32 -> 8x8
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedChessPieceClassifier(nn.Module):
    """Enhanced chess piece classifier with minimal complexity increase"""

    def __init__(self, num_classes=13, use_grayscale=DEFAULT_USE_GRAYSCALE, dropout_rate=0.3):
        super(EnhancedChessPieceClassifier, self).__init__()
        input_channels = 1 if use_grayscale else 3

        # Feature extraction with residual connections and batch norm
        self.conv_layers = nn.Sequential(
            # First convolutional block
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second convolutional block with residual-like connection
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third convolutional block - deeper feature extraction
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Additional conv block for better feature learning
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),  # Light spatial dropout
        )

        # Global Average Pooling instead of flattening full feature maps
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Enhanced classifier with intermediate layer
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate/2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.global_pool(x)  # Global average pooling
        x = self.fc_layers(x)
        return x


class UltraEnhancedChessPieceClassifier(nn.Module):
    """Most advanced version with attention and ensemble-like features"""

    def __init__(self, num_classes=13, use_grayscale=DEFAULT_USE_GRAYSCALE, dropout_rate=0.3):
        super(UltraEnhancedChessPieceClassifier, self).__init__()
        input_channels = 1 if use_grayscale else 3

        # Feature extraction backbone
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
        )

        # Simple attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Multiple pooling strategies
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))
        
        # Enhanced classifier
        self.fc_layers = nn.Sequential(
            nn.Linear(256, 128),  # 128*2 from avg+max pooling
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate/2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        # Apply attention
        attention_weights = self.attention(x)
        x = x * attention_weights
        
        # Multiple pooling
        avg_pool = self.global_avg_pool(x)
        max_pool = self.global_max_pool(x)
        
        # Concatenate different pooling results
        x = torch.cat([avg_pool, max_pool], dim=1)
        x = x.view(x.size(0), -1)
        
        x = self.fc_layers(x)
        return x

