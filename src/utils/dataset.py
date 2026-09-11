"""
Shared PyTorch Dataset for loading Animal-80 images from manifest.csv.

Used by every model training script (baseline, efficient CNN, modern CNN)
so that all models load and preprocess data identically, keeping the
model comparison fair.

Usage:
    from src.utils.dataset import AnimalDataset
    from src.utils.transforms import get_train_transform, get_eval_transform

    train_dataset = AnimalDataset("data/manifest.csv", "train", get_train_transform())
    val_dataset = AnimalDataset("data/manifest.csv", "val", get_eval_transform())
    test_dataset = AnimalDataset("data/manifest.csv", "test", get_eval_transform())
"""
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class AnimalDataset(Dataset):
    def __init__(self, manifest_path, split, transform):
        """
        manifest_path: path to data/manifest.csv
        split: one of "train", "val", "test"
        transform: a torchvision transform, e.g. from src.utils.transforms
        """
        df = pd.read_csv(manifest_path)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.transform = transform
        self.classes = sorted(self.df["label"].unique())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["filepath"]).convert("RGB")
        img = self.transform(img)
        label = self.class_to_idx[row["label"]]
        return img, label