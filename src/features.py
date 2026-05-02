import re
from typing import List, Tuple

import emoji
import numpy as np
import pandas as pd


BLACKLIST_COLUMNS = [
    "dialog",
    "p0_messages",
    "p1_messages",
    "label",
    "p0_bot",
    "p1_bot",
    "dialog_id",
]


def count_emoji(text: str) -> int:
    return sum(1 for char in str(text) if char in emoji.EMOJI_DATA)


def add_dialog_length(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dialog_length"] = df["dialog"].apply(len)
    return df


def add_messages_by_persons(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["p0_messages"] = df["dialog"].apply(lambda x: [msg["text"] for msg in x if msg["person"] == "0"])
    df["p1_messages"] = df["dialog"].apply(lambda x: [msg["text"] for msg in x if msg["person"] == "1"])
    return df


def _safe_mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["p0_text"] = df["p0_messages"].apply(lambda x: " ".join(x))
    df["p1_text"] = df["p1_messages"].apply(lambda x: " ".join(x))

    for prefix in ["p0", "p1"]:
        messages_col = f"{prefix}_messages"
        text_col = f"{prefix}_text"

        df[f"{prefix}_emoji_count"] = df[text_col].apply(count_emoji)
        df[f"{prefix}_emoji_percent"] = df[messages_col].apply(
            lambda messages: sum(count_emoji(msg) > 0 for msg in messages) / len(messages) if messages else 0
        )
        df[f"{prefix}_mean_length"] = df[messages_col].apply(lambda messages: _safe_mean([len(msg) for msg in messages]))
        df[f"{prefix}_special_count"] = df[text_col].apply(
            lambda text: sum(text.count(char) for char in ["?", "!"]) / len(text) if text else 0
        )
        df[f"{prefix}_first_capital_percent"] = df[messages_col].apply(
            lambda messages: sum(len(msg) > 0 and msg[0].isupper() for msg in messages) / len(messages) if messages else 0
        )
        df[f"{prefix}_capital_percent"] = df[text_col].apply(
            lambda text: sum(char.isupper() for char in text) / len(text) if text else 0
        )
        df[f"{prefix}_mean_words_length"] = df[text_col].apply(
            lambda text: _safe_mean([len(word) for word in text.split()]) if text else 0
        )
        df[f"{prefix}_mean_words_count"] = df[messages_col].apply(
            lambda messages: _safe_mean([len(msg.split()) for msg in messages]) if messages else 0
        )

    return df.drop(columns=["p0_text", "p1_text"])


def add_echo_bot_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def echo_count(dialog):
        p0_echo_count = 0
        p1_echo_count = 0
        prev_text = None

        for msg in dialog:
            text = msg["text"]
            person = msg["person"]

            if prev_text is not None and text == prev_text:
                if person == "0":
                    p0_echo_count += 1
                elif person == "1":
                    p1_echo_count += 1

            prev_text = text

        return p0_echo_count, p1_echo_count

    temp = df["dialog"].apply(echo_count)
    df["p0_echo_index"] = temp.apply(lambda x: x[0]) / df["p0_messages"].apply(lambda x: len(x) - 1 if len(x) > 1 else 1)
    df["p1_echo_index"] = temp.apply(lambda x: x[1]) / df["p1_messages"].apply(lambda x: len(x) if len(x) > 0 else 1)

    return df


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_dialog_length(df)
    df = add_messages_by_persons(df)
    df = add_text_features(df)
    df = add_echo_bot_index(df)
    return df


def select_model_features(df: pd.DataFrame) -> pd.DataFrame:
    cols = [col for col in df.columns if col not in BLACKLIST_COLUMNS]
    return df[cols]


def split_labels(y: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Преобразует общую метку в две бинарные задачи.

    label=1: participant 0 — бот, participant 1 — человек
    label=0: participant 0 — человек, participant 1 — бот
    label=2: оба человека
    """
    p0_bot = (y == 1).astype(int)
    p1_bot = (y == 0).astype(int)
    return p0_bot, p1_bot


def get_texts_p0(df: pd.DataFrame) -> List[str]:
    return df["p0_messages"].apply(lambda x: "\n".join(x)).tolist()


def get_texts_p1(df: pd.DataFrame) -> List[str]:
    return df["p1_messages"].apply(lambda x: "\n".join(x)).tolist()


def get_bot_texts(df: pd.DataFrame) -> List[str]:
    p0_texts = get_texts_p0(df[df["p0_bot"] == 1])
    p1_texts = get_texts_p1(df[df["p1_bot"] == 1])
    return p0_texts + p1_texts


def get_human_texts(df: pd.DataFrame) -> List[str]:
    p0_texts = get_texts_p0(df[df["p0_bot"] == 0])
    p1_texts = get_texts_p1(df[df["p1_bot"] == 0])
    return p0_texts + p1_texts


def preprocess_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()
