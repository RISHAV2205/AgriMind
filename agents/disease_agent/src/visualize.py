
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from agents.disease_agent.src.dataset import create_datasets, CLASS_NAMES


def denormalize(image):
    """
    Reverse ImageNet normalization so the image
    looks normal when displayed.
    """

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    image = image * std + mean

    return torch.clamp(image, 0, 1)


def visualize_training_images(num_images=8):
    """
    Display randomly augmented training images.
    """

    train_dataset, _, _ = create_datasets()

    # Create a figure
    fig, axes = plt.subplots(
        2,
        4,
        figsize=(14, 7)
    )

    axes = axes.flatten()

    for i in range(num_images):

        # Get one training sample
        image, label = train_dataset[i]

        # Undo normalization
        image = denormalize(image)

        # Convert:
        # [C, H, W] -> [H, W, C]
        image = image.permute(1, 2, 0)

        # Display image
        axes[i].imshow(image)

        # Display class name
        axes[i].set_title(
            CLASS_NAMES[label],
            fontsize=10
        )

        axes[i].axis("off")

    plt.suptitle(
        "AgriMind - Training Dataset Visualization",
        fontsize=16
    )

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":

    print("Loading training dataset...")

    visualize_training_images()

    print("Visualization completed.")

