from torchvision import transforms


# Image size expected by our models resnet
IMAGE_SIZE = 224

def get_train_transforms():
    """
    Transformations applied to training images.

    Training images receive augmentation so that
    the model learns to generalize better.
    """

    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

        # Data augmentation
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),

        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
        ),

        # Convert PIL image to PyTorch tensor
        transforms.ToTensor(),

        # ImageNet normalization
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_eval_transforms():
    """
    Transformations applied to validation and test images.

    No random augmentation is used here.
    """

    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])