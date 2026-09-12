from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from agents.disease_agent.src.preprocessing import (
    get_train_transforms,
    get_eval_transforms
)


CSV_PATH = Path(
    r"D:\AgriMind\agents\disease_agent\data\processed\dataset_split.csv"
)

DATASET_ROOT = Path(
    r"D:\AgriMind\agents\disease_agent\data\raw\tomato_5_classes"
)

BATCH_SIZE = 32

NUM_WORKERS = 0

CLASS_NAMES = [
    "Tomato_Early_blight",
    "Tomato_healthy",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
]


class TomatoDiseaseDataset(Dataset):

    def __init__(self, dataframe, transform=None):

        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):

        row = self.dataframe.iloc[index]
        image_path = DATASET_ROOT / row["image_path"]
        label = int(row["label"])

        # Open image
        image = Image.open(image_path).convert("RGB")
        # print(image)

        # Apply transformations
        if self.transform is not None:
            image = self.transform(image)

        return image, label


# --------------------------------------------------
# Create datasets
# --------------------------------------------------

def create_datasets():

    # Read CSV
    df = pd.read_csv(CSV_PATH)

    # Separate splits
    train_df = df[df["split"] == "train"].copy()

    val_df = df[df["split"] == "val"].copy()

    test_df = df[df["split"] == "test"].copy()

    # Create datasets
    train_dataset = TomatoDiseaseDataset(
        train_df,
        transform=get_train_transforms()
    )

    val_dataset = TomatoDiseaseDataset(
        val_df,
        transform=get_eval_transforms()
    )

    test_dataset = TomatoDiseaseDataset(
        test_df,
        transform=get_eval_transforms()
    )

    return train_dataset, val_dataset, test_dataset




def create_dataloaders():

    train_dataset, val_dataset, test_dataset = create_datasets()
    # because we don't want the model to see training examples in the same order every epoch.
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    return train_loader, val_loader, test_loader



if __name__ == "__main__":

    train_loader, val_loader, test_loader = create_dataloaders()

    print("Dataset loaded successfully! ✅")
    

    print(f"Training samples:   {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples:       {len(test_loader.dataset)}")

    print(f"\nNumber of classes: {len(CLASS_NAMES)}")
    print(f"Classes: {CLASS_NAMES}")

    # Get one batch
    images, labels = next(iter(train_loader))

    print("\nFirst training batch:")
    print(f"Image tensor shape: {images.shape}")
    print(f"Label tensor shape: {labels.shape}")

    print(f"\nLabels in first batch:")
    print(labels)