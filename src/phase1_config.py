from pathlib import Path

SEED = 42

DATA_DIR = Path("dataset")
OUTPUT_DIR = Path("outputs")
PLOTS_DIR = OUTPUT_DIR / "plots"
SPLITS_DIR = Path("data_splits")

IMAGE_SIZE = 64
BATCH_SIZE = 32

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
