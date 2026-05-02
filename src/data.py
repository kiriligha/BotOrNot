import json
from pathlib import Path
from typing import Tuple

import pandas as pd

from config import TRAIN_JSON, TEST_JSON, YTRAIN_CSV, YTEST_CSV, SAMPLE_SUBMISSION_CSV


LABEL_MAP = {
    "01": 0, 
    "10": 1, 
    "00": 2, 
}


def _check_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}. Положи данные Kaggle в папку data/raw/.")


def load_raw_data() -> Tuple[dict, dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Загружает исходные файлы конкурса.

    Ожидаемые файлы в data/raw/:
    - train.json
    - test.json
    - ytrain.csv
    - ytest.csv
    - sample_submission.csv
    """
    for path in [TRAIN_JSON, TEST_JSON, YTRAIN_CSV, YTEST_CSV, SAMPLE_SUBMISSION_CSV]:
        _check_file(path)

    with open(TRAIN_JSON, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    with open(TEST_JSON, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    ytrain = pd.read_csv(YTRAIN_CSV)
    ytest = pd.read_csv(YTEST_CSV)
    sample_submission = pd.read_csv(SAMPLE_SUBMISSION_CSV)

    return train_data, test_data, ytrain, ytest, sample_submission


def build_train_dataframe(train_data: dict, ytrain: pd.DataFrame) -> pd.DataFrame:
    dataset = []

    for dialog_id, dialog in train_data.items():
        mask = ytrain["dialog_id"] == dialog_id
        labels = (
            ytrain.loc[mask, ["participant_index", "is_bot"]]
            .sort_values("participant_index")
            ["is_bot"]
            .astype(int)
            .tolist()
        )

        if len(labels) != 2:
            raise ValueError(f"Для dialog_id={dialog_id} ожидалось 2 метки, найдено: {len(labels)}")

        p0_bot = labels[0]
        p1_bot = labels[1]
        combined_label = LABEL_MAP[f"{p0_bot}{p1_bot}"]

        prepared_dialog = [
            {
                "person": str(message.get("participant_index", message.get("person"))),
                "text": str(message.get("text", "")),
            }
            for message in dialog
        ]

        dataset.append(
            {
                "dialog_id": dialog_id,
                "dialog": prepared_dialog,
                "label": combined_label,
                "p0_bot": p0_bot,
                "p1_bot": p1_bot,
            }
        )

    return pd.DataFrame(dataset)


def build_test_dataframe(test_data: dict) -> pd.DataFrame:
    dataset = []

    for dialog_id, dialog in test_data.items():
        prepared_dialog = [
            {
                "person": str(message.get("participant_index", message.get("person"))),
                "text": str(message.get("text", "")),
            }
            for message in dialog
        ]

        dataset.append({"dialog_id": dialog_id, "dialog": prepared_dialog})

    return pd.DataFrame(dataset)


def load_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_data, test_data, ytrain, ytest, sample_submission = load_raw_data()
    df_train = build_train_dataframe(train_data, ytrain)
    df_test = build_test_dataframe(test_data)
    return df_train, df_test, ytest, sample_submission
