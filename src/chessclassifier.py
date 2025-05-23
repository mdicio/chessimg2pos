import torch.nn as nn

class ChessPieceClassifier(nn.Module):
    """Chess piece classifier model"""

    def __init__(self, num_classes=13, use_grayscale=True):
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
