import torch
import torch.nn as nn

class BaselineCNN(nn.Module):
    """
    Simple CNN baseline for tomato leaf disease classification.

    Input:
        [batch_size, 3, 224, 224]

    Output:
        [batch_size, 5]
    """

    def __init__(self, num_classes=5):
        super().__init__()
        # Feature extraction
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # 224 -> 112


            # Block 2
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # 112 -> 56


            # Block 3
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # 56 -> 28


            # Block 4
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # 28 -> 14
        )

        # Adaptive pooling avoids hard-coding
        # the spatial dimensions.
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification head
        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(256, 128),

            nn.ReLU(),

            nn.Dropout(p=0.5),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):

        x = self.features(x)

        x = self.pool(x)

        x = self.classifier(x)

        return x


if __name__ == "__main__":

    # Test model
    model = BaselineCNN(num_classes=5)

    print(model)

    # Dummy input
    x = torch.randn(4, 3, 224, 224)

    # Forward pass
    output = model(x)

    print("\nInput shape:")
    print(x.shape)

    print("\nOutput shape:")
    print(output.shape)
