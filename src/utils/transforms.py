"""
Shared image transforms for training, validation, and test data.

Applies the dataset-derived normalization values (config.NORMALIZATION_MEAN,
config.NORMALIZATION_STD) via standard resize/normalize pipelines, so all
model training code uses consistent preprocessing.

Usage (PyTorch):
    from src.utils.transforms import get_train_transform, get_eval_transform
    train_tf = get_train_transform()
    eval_tf = get_eval_transform()
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils import config

try:
    from torchvision import transforms
    _HAS_TORCHVISION = True
except ImportError:
    _HAS_TORCHVISION = False


def get_train_transform():
    """
    Transform for training data.

    Heavy augmentation (rotation, shear, zoom, translation) is already baked
    into the saved images by src/preprocessing/augment.py for rare classes,
    so this only applies light on-the-fly augmentation (horizontal flip)
    plus resize/normalize, to avoid double augmentation.
    """
    if not _HAS_TORCHVISION:
        raise ImportError("torchvision is required for get_train_transform(). "
                           "Install with: pip install torchvision")
    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORMALIZATION_MEAN, std=config.NORMALIZATION_STD),
    ])


def get_eval_transform():
    """Transform for validation/test data: resize and normalize only, no augmentation."""
    if not _HAS_TORCHVISION:
        raise ImportError("torchvision is required for get_eval_transform(). "
                           "Install with: pip install torchvision")
    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORMALIZATION_MEAN, std=config.NORMALIZATION_STD),
    ])


def get_normalization_values():
    """Framework-agnostic access to the raw mean/std (e.g. for TensorFlow/Keras)."""
    return config.NORMALIZATION_MEAN, config.NORMALIZATION_STD


if __name__ == "__main__":
    mean, std = get_normalization_values()
    print(f"Normalization mean: {mean}")
    print(f"Normalization std:  {std}")
    if _HAS_TORCHVISION:
        print("torchvision available: get_train_transform() / get_eval_transform() ready to use.")
    else:
        print("torchvision not installed. Install with: pip install torchvision")