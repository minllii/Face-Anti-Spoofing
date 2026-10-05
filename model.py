import torch
import torch.nn as nn


class SimpleCNN(nn.Module):

  def __init__(self):
    super(SimpleCNN, self).__init__()

    # Feature Extractor: Convolutional layers process the image pixels
    self.features = nn.Sequential(
        # Block 1
        nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1
        ),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),  # Cuts image size in half (60x60)
        # Block 2
        nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1
        ),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),  # Cuts image size in half (30x30)
        # Block 3
        nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        ),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),  # Cuts image size in half (15x15)
    )

    # Classifier: Fully connected layers that make the final decision
    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(64 * 16 * 16, 128),  
        # 16x16 is the spatial size of a 128x128 image after 3 max-pooling steps
        nn.ReLU(),
        nn.Dropout(0.5),  # Drops 50% of connections randomly to prevent overfitting
        nn.Linear(128, 2),  # Output layer: 2 classes (Spoof or Live)
    )

  def forward(self, x):
    # Passes the image through the convolutional features, then the classifier
    x = self.features(x)
    x = self.classifier(x)
    return x