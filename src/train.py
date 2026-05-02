import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import lightgbm as lgb
import xgboost as xgb
import catboost as cat

from config import SEED, N_SPLITS, MODELS_DIR, DATA_PROCESSED_DIR
from data import load_datasets
from features import (
    make_features,
    select_model_features,
    split_labels,
    get_texts_p0,
    get_texts_p1,
    get_bot_texts,
    get_human_texts,
    preprocess_text,
)
from utils import set_all_seeds, ensure_dirs


def get_base_models():
    return {
        "lgbm": lgb.LGBMClassifier(verbose=-1, random_state=SEED),
        "xgb": xgb.XGBClassifier(eval_metric="logloss", random_state=SEED),
        "cat": cat.CatBoostClassifier(verbose=0, random_state=SEED),
        "rf": RandomForestClassifier(random_state=SEED),
        "lr": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED)),
    }


def add_probs_to_df(df, probs_p0, probs_p1, model_name, indexes=None):
    if indexes is None:
        indexes = df.index.tolist()

    for col in [f"p0_prob_{model_name}", f"p1_prob_{model_name}"]:
        if col not in df.columns:
            df[col] = np.nan

    df.loc[df.index[indexes], f"p0_prob_{model_name}"] = probs_p0[:, 1]
    df.loc[df.index[indexes], f"p1_prob_{model_name}"] = probs_p1[:, 1]


def train_base_models(X_train, y_train, X_test):
    y_p0, y_p1 = split_labels(y_train)

    X_train_meta = X_train.copy()
    X_test_meta = X_test.copy()

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    model_names = list(get_base_models().keys())

    test_probs_p0 = {name: np.zeros((len(X_test), 2)) for name in model_names}
    test_probs_p1 = {name: np.zeros((len(X_test), 2)) for name in model_names}
    metrics = []

    for fold, (train_idx, val_idx) in tqdm(enumerate(kf.split(X_train)), total=N_SPLITS, desc="Базовые модели"):
        X_tr = X_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]
        y_p0_tr = y_p0.iloc[train_idx]
        y_p0_val = y_p0.iloc[val_idx]
        y_p1_tr = y_p1.iloc[train_idx]
        y_p1_val = y_p1.iloc[val_idx]

        models_p0 = get_base_models()
        models_p1 = get_base_models()

        fold_metrics = {"fold": fold + 1}

        for name in model_names:
            model_p0 = models_p0[name]
            model_p1 = models_p1[name]

            model_p0.fit(X_tr, y_p0_tr)
            model_p1.fit(X_tr, y_p1_tr)

            val_probs_p0 = model_p0.predict_proba(X_val)
            val_probs_p1 = model_p1.predict_proba(X_val)

            add_probs_to_df(X_train_meta, val_probs_p0, val_probs_p1, name, val_idx)

            test_probs_p0[name] += model_p0.predict_proba(X_test) / N_SPLITS
            test_probs_p1[name] += model_p1.predict_proba(X_test) / N_SPLITS

            fold_metrics[f"{name}_p0_acc"] = accuracy_score(y_p0_val, model_p0.predict(X_val))
            fold_metrics[f"{name}_p1_acc"] = accuracy_score(y_p1_val, model_p1.predict(X_val))
            fold_metrics[f"{name}_p0_logloss"] = log_loss(y_p0_val, val_probs_p0)
            fold_metrics[f"{name}_p1_logloss"] = log_loss(y_p1_val, val_probs_p1)

        metrics.append(fold_metrics)

    for name in model_names:
        add_probs_to_df(X_test_meta, test_probs_p0[name], test_probs_p1[name], name)

    return X_train_meta, X_test_meta, pd.DataFrame(metrics)


def get_tfidf_probs(df, vectorizer, model):
    texts_p0 = [preprocess_text(text) for text in get_texts_p0(df)]
    texts_p1 = [preprocess_text(text) for text in get_texts_p1(df)]

    vec_p0 = vectorizer.transform(texts_p0)
    vec_p1 = vectorizer.transform(texts_p1)

    return model.predict_proba(vec_p0), model.predict_proba(vec_p1)


def train_tfidf_model(df_train_features, df_test_features, X_train_meta, X_test_meta):
    y_p0, y_p1 = split_labels(df_train_features["label"])
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    test_probs_p0 = np.zeros((len(df_test_features), 2))
    test_probs_p1 = np.zeros((len(df_test_features), 2))
    metrics = []

    for fold, (train_idx, val_idx) in tqdm(enumerate(kf.split(df_train_features)), total=N_SPLITS, desc="TF-IDF"):
        train_fold = df_train_features.iloc[train_idx].copy()
        val_fold = df_train_features.iloc[val_idx].copy()

        bot_texts = [preprocess_text(text) for text in get_bot_texts(train_fold)]
        human_texts = [preprocess_text(text) for text in get_human_texts(train_fold)]

        X_text = bot_texts + human_texts
        y_text = [1] * len(bot_texts) + [0] * len(human_texts)

        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(2, 3), analyzer="char")
        X_vec = vectorizer.fit_transform(X_text)

        model = LogisticRegression(max_iter=1000, random_state=SEED)
        model.fit(X_vec, y_text)

        val_probs_p0, val_probs_p1 = get_tfidf_probs(val_fold, vectorizer, model)
        add_probs_to_df(X_train_meta, val_probs_p0, val_probs_p1, "tfidf", val_idx)

        test_fold_probs_p0, test_fold_probs_p1 = get_tfidf_probs(df_test_features, vectorizer, model)
        test_probs_p0 += test_fold_probs_p0 / N_SPLITS
        test_probs_p1 += test_fold_probs_p1 / N_SPLITS

        metrics.append(
            {
                "fold": fold + 1,
                "p0_acc": accuracy_score(y_p0.iloc[val_idx], val_probs_p0[:, 1] > 0.5),
                "p1_acc": accuracy_score(y_p1.iloc[val_idx], val_probs_p1[:, 1] > 0.5),
                "p0_logloss": log_loss(y_p0.iloc[val_idx], val_probs_p0),
                "p1_logloss": log_loss(y_p1.iloc[val_idx], val_probs_p1),
            }
        )

    add_probs_to_df(X_test_meta, test_probs_p0, test_probs_p1, "tfidf")
    return X_train_meta, X_test_meta, pd.DataFrame(metrics)


def train_meta_models(X_train_meta, y_train):
    prob_cols = [col for col in X_train_meta.columns if col.startswith("p0_prob_") or col.startswith("p1_prob_")]
    X = X_train_meta[prob_cols].copy()

    y_p0, y_p1 = split_labels(y_train)
    X_tr, X_val, y_p0_tr, y_p0_val, y_p1_tr, y_p1_val = train_test_split(
        X, y_p0, y_p1, test_size=0.2, random_state=SEED, stratify=y_train
    )

    meta_model_p0 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED))
    meta_model_p1 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED))

    meta_model_p0.fit(X_tr, y_p0_tr)
    meta_model_p1.fit(X_tr, y_p1_tr)

    probs_p0 = meta_model_p0.predict_proba(X_val)
    probs_p1 = meta_model_p1.predict_proba(X_val)

    print("\nМетрики meta-моделей:")
    print(f"P0 accuracy:  {accuracy_score(y_p0_val, probs_p0[:, 1] > 0.5):.4f}")
    print(f"P1 accuracy:  {accuracy_score(y_p1_val, probs_p1[:, 1] > 0.5):.4f}")
    print(f"P0 log loss:  {log_loss(y_p0_val, probs_p0):.4f}")
    print(f"P1 log loss:  {log_loss(y_p1_val, probs_p1):.4f}")

    meta_model_p0.fit(X, y_p0)
    meta_model_p1.fit(X, y_p1)

    return meta_model_p0, meta_model_p1, prob_cols


def main():
    set_all_seeds(SEED)
    ensure_dirs(MODELS_DIR, DATA_PROCESSED_DIR)

    print("Загрузка данных...")
    df_train, df_test, ytest, sample_submission = load_datasets()
    print(f"Train dialogs: {len(df_train)}")
    print(f"Test dialogs:  {len(df_test)}")
    print(f"ytest rows:    {len(ytest)}")

    print("Создание признаков...")
    df_train_features = make_features(df_train)
    df_test_features = make_features(df_test)

    X_train = select_model_features(df_train_features)
    X_test = select_model_features(df_test_features)
    X_test = X_test[X_train.columns]
    y_train = df_train_features["label"].copy()

    print("Обучение базовых моделей...")
    X_train_meta, X_test_meta, base_metrics = train_base_models(X_train, y_train, X_test)

    print("Обучение TF-IDF модели...")
    X_train_meta, X_test_meta, tfidf_metrics = train_tfidf_model(
        df_train_features, df_test_features, X_train_meta, X_test_meta
    )

    print("Обучение meta-моделей...")
    meta_model_p0, meta_model_p1, prob_cols = train_meta_models(X_train_meta, y_train)

    artifacts = {
        "meta_model_p0": meta_model_p0,
        "meta_model_p1": meta_model_p1,
        "prob_cols": prob_cols,
        "test_meta_features": X_test_meta[prob_cols],
        "df_test": df_test_features[["dialog_id", "dialog"]].copy(),
        "sample_submission": sample_submission,
    }

    joblib.dump(artifacts, MODELS_DIR / "stacking_artifacts.joblib")
    base_metrics.to_csv(DATA_PROCESSED_DIR / "base_cv_metrics.csv", index=False)
    tfidf_metrics.to_csv(DATA_PROCESSED_DIR / "tfidf_cv_metrics.csv", index=False)

    print(f"\nАртефакты сохранены в: {MODELS_DIR / 'stacking_artifacts.joblib'}")


if __name__ == "__main__":
    main()
