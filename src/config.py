from pathlib import Path

SEED = 42
N_SPLITS = 5

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
SUBMISSIONS_DIR = ROOT_DIR / "submissions"
REPORTS_DIR = ROOT_DIR / "reports"

TRAIN_JSON = DATA_RAW_DIR / "train.json"
TEST_JSON = DATA_RAW_DIR / "test.json"
YTRAIN_CSV = DATA_RAW_DIR / "ytrain.csv"
YTEST_CSV = DATA_RAW_DIR / "ytest.csv"
SAMPLE_SUBMISSION_CSV = DATA_RAW_DIR / "sample_submission.csv"

SUBMISSION_PATH = SUBMISSIONS_DIR / "my_best_submission.csv"
