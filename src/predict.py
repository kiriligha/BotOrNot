import joblib
import pandas as pd

from config import MODELS_DIR, SUBMISSIONS_DIR, SUBMISSION_PATH
from utils import ensure_dirs


def main():
    ensure_dirs(SUBMISSIONS_DIR)

    artifacts_path = MODELS_DIR / "stacking_artifacts.joblib"
    if not artifacts_path.exists():
        raise FileNotFoundError("Сначала обучи модель: python src/train.py")

    artifacts = joblib.load(artifacts_path)

    meta_model_p0 = artifacts["meta_model_p0"]
    meta_model_p1 = artifacts["meta_model_p1"]
    prob_cols = artifacts["prob_cols"]
    X_test_meta = artifacts["test_meta_features"][prob_cols]
    df_test = artifacts["df_test"]
    submission = artifacts["sample_submission"].copy()

    probs_p0 = meta_model_p0.predict_proba(X_test_meta)[:, 1]
    probs_p1 = meta_model_p1.predict_proba(X_test_meta)[:, 1]

    dialog_predictions = df_test[["dialog_id"]].copy()
    dialog_predictions["is_bot_0"] = probs_p0
    dialog_predictions["is_bot_1"] = probs_p1

    pred_map = {}
    for _, row in dialog_predictions.iterrows():
        pred_map[f"{row['dialog_id']}_0"] = row["is_bot_0"]
        pred_map[f"{row['dialog_id']}_1"] = row["is_bot_1"]

    submission["is_bot"] = submission["ID"].map(pred_map).astype("float32")

    if submission["is_bot"].isna().any():
        missing = submission.loc[submission["is_bot"].isna(), "ID"].head().tolist()
        raise ValueError(f"Для части ID не нашлись предсказания. Примеры: {missing}")

    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission сохранён: {SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
