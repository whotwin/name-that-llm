"""用 pandas 读取比赛数据集的 train.csv 和 test.csv。"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "name-that-llm"


def read_train() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "train.csv")


def read_test() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "test.csv")


if __name__ == "__main__":
    for name, df in [("train", read_train()), ("test", read_test())]:
        print(f"=== {name} ===")
        model_list = df['model'].dropna().unique()
        print(f"shape: {df.shape}")
        print(f"columns: {list(df.columns)}")
        print(df.head(50))
        print(model_list.tolist())
        print()
        break
