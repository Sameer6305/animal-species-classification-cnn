import random
import sys
from pathlib import Path
import pandas as pd
from PIL import Image, ImageOps

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils import config


def random_augment(img: Image.Image) -> Image.Image:
    #rotation
    angle = random.uniform(-25, 25)
    img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=(0, 0, 0))

    #mirror
    if random.random() < 0.5:
        img = ImageOps.mirror(img)

    if random.random() < 0.5:  # zoom
        w, h = img.size
        zf = random.uniform(0.8, 0.95)
        nw, nh = int(w * zf), int(h * zf)
        left, top = random.randint(0, w - nw), random.randint(0, h - nh)
        img = img.crop((left, top, left + nw, top + nh)).resize((w, h), Image.BILINEAR)

    if random.random() < 0.5:  # translation
        w, h = img.size
        shift = int(0.1 * w)
        dx, dy = random.randint(-shift, shift), random.randint(-shift, shift)
        img = ImageOps.expand(img, border=shift, fill=(0, 0, 0))
        img = img.crop((shift + dx, shift + dy, shift + dx + w, shift + dy + h))

    if random.random() < 0.4:  # shear
        import math
        shear_factor = random.uniform(-0.15, 0.15)
        w, h = img.size
        img = img.transform(
            (w, h), Image.AFFINE,
            (1, shear_factor, -shear_factor * h / 2, 0, 1, 0),
            resample=Image.BILINEAR, fillcolor=(0, 0, 0)
        )

    return img


def main():
    df = pd.read_csv(config.MANIFEST_PATH)
    train_df = df[df["split"] == "train"]
    class_counts = train_df.groupby("label").size()
    rare_classes = class_counts[class_counts < config.RARE_CLASS_THRESHOLD]

    print(f"Rare classes to augment ({len(rare_classes)}):\n{rare_classes}\n")

    new_rows = []
    for class_name, count in rare_classes.items():
        needed = config.RARE_CLASS_TARGET - count
        if needed <= 0:
            continue

        class_rows = train_df[train_df["label"] == class_name]
        out_dir = config.PROCESSED_DIR / "train" / class_name
        out_dir.mkdir(parents=True, exist_ok=True)

        for i in range(needed):
            src_row = class_rows.sample(1).iloc[0]
            src_path = config.PROJECT_ROOT / src_row["filepath"]
            try:
                img = Image.open(src_path).convert("RGB")
            except Exception as e:
                print(f"  [warn] {src_path}: {e}")
                continue

            aug_img = random_augment(img)
            out_path = out_dir / f"aug_{i}_{src_path.stem}.jpg"
            aug_img.save(out_path, quality=95)

            new_rows.append({
                "filepath": str(out_path.relative_to(config.PROJECT_ROOT)),
                "label": class_name,
                "split": "train",
                "source_image": src_row["source_image"],
            })

        print(f"  {class_name}: {count} -> {count + needed}")

    if new_rows:
        full_df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        full_df.to_csv(config.MANIFEST_PATH, index=False)
        print(f"\nAdded {len(new_rows)} augmented images. Manifest updated.")


if __name__ == "__main__":
    main()