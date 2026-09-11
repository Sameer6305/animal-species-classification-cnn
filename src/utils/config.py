from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest.csv"

TRAIN_DIR = RAW_DIR / "train"
TEST_DIR = RAW_DIR / "test"

IMG_SIZE = 224

VAL_FRACTION = 0.15
MIN_VAL_PER_CLASS = 2      # even tiny classes get at least this many in val

RARE_CLASS_THRESHOLD = 100  # train classes below this count are "rare"
RARE_CLASS_TARGET = 150     # augment rare classes up to roughly this many

# Computed once via dataset_stats.compute_normalization_stats(), then hardcoded here
# for consistent use across all model training scripts.
NORMALIZATION_MEAN = [0.46290031, 0.44127762, 0.38908589]
NORMALIZATION_STD = [0.26581346, 0.2520165, 0.25977104]