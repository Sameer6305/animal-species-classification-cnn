import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils import config


def assign_val_split(df_class: pd.DataFrame):
    n = len(df_class)
    n_val = max(config.MIN_VAL_PER_CLASS, round(n * config.VAL_FRACTION))
    n_val = min(n_val, n - 1) if n > 1 else 0  # always leave at least 1 for train
    shuffled = df_class.sample(frac=1, random_state=42).reset_index(drop=True)
    labels = ["val"] * n_val + ["train"] * (n - n_val)
    return shuffled, labels


def main():
    raw_manifest_path = config.PROJECT_ROOT / "data" / "processed_manifest_raw.csv"
    df = pd.read_csv(raw_manifest_path)

    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()
    test_df["final_split"] = "test"

    parts = []
    for class_name, group in train_df.groupby("label"):
        shuffled, labels = assign_val_split(group)
        shuffled["final_split"] = labels
        parts.append(shuffled)

    final_train_df = pd.concat(parts, ignore_index=True)
    final_df = pd.concat([final_train_df, test_df], ignore_index=True)
    final_df = final_df[["filepath", "label", "final_split", "source_image"]].rename(
        columns={"final_split": "split"}
    )

    final_df.to_csv(config.MANIFEST_PATH, index=False)
    print("Split summary:")
    print(final_df.groupby("split").size())
    print(f"\nFinal manifest: {config.MANIFEST_PATH}")


if __name__ == "__main__":
    main()