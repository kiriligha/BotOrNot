import matplotlib.pyplot as plt

from config import REPORTS_DIR
from data import load_datasets
from features import make_features
from utils import ensure_dirs


def main():
    ensure_dirs(REPORTS_DIR / "figures")

    df_train, _, _, _ = load_datasets()
    df_train = make_features(df_train)

    plt.figure(figsize=(8, 5))
    df_train["label"].value_counts(normalize=True).sort_index().plot(kind="bar")
    plt.title("Распределение классов")
    plt.xlabel("Класс")
    plt.ylabel("Доля")
    plt.xticks(ticks=[0, 1, 2], labels=["P0 человек, P1 бот", "P0 бот, P1 человек", "Оба люди"], rotation=15)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "figures" / "label_distribution.png")

    print(f"График сохранён: {REPORTS_DIR / 'figures' / 'label_distribution.png'}")


if __name__ == "__main__":
    main()
