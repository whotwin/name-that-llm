"""用 TF-IDF 从 train/test 文本提取特征，保存稀疏矩阵和 vectorizer 到 artifacts/。

两套特征：
- 词级 n-gram（1,2）：捕捉用词和短语习惯。不删停用词——功能词恰恰是"谁写的"的线索。
- 字符级 n-gram（char_wb, 2,5）：对拼写、标点、词尾等风格细节更敏感。

只提取特征，不训练分类器。
"""

from pathlib import Path

import joblib
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from read_files import read_train, read_test

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"


def build_vectorizers():
    word_vec = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), sublinear_tf=True, min_df=2
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2
    )
    return word_vec, char_vec


def main():
    train = read_train()
    test = read_test()

    word_vec, char_vec = build_vectorizers()
    # 各自 fit 在 train 文本上
    X_word = word_vec.fit_transform(train["text"])
    X_char = char_vec.fit_transform(train["text"])
    X = sp.hstack([X_word, X_char], format="csr")

    # 测试集 transform（目前只有 3 行占位样例，仅验证形状）
    X_test_word = word_vec.transform(test["text"])
    X_test_char = char_vec.transform(test["text"])
    X_test = sp.hstack([X_test_word, X_test_char], format="csr")

    # 持久化
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    sp.save_npz(ARTIFACTS_DIR / "train_features.npz", X)
    sp.save_npz(ARTIFACTS_DIR / "test_features.npz", X_test)
    train["model"].to_csv(ARTIFACTS_DIR / "train_labels.csv", index=False)
    joblib.dump(word_vec, ARTIFACTS_DIR / "tfidf_word.joblib")
    joblib.dump(char_vec, ARTIFACTS_DIR / "tfidf_char.joblib")

    # 打印概况
    print(f"word matrix: {X_word.shape}")
    print(f"char matrix: {X_char.shape}")
    print(f"merged train matrix: {X.shape}")
    print(f"merged test matrix:  {X_test.shape}")
    print(f"labels: {train['model'].value_counts().to_dict()}")

    # 词级 tf-idf 权重最高的 10 个特征（人工看一眼是否是有区分度的风格词）
    mean_scores = np.asarray(X_word.mean(axis=0)).ravel()
    top_idx = np.argsort(mean_scores)[::-1][:10]
    feature_names = word_vec.get_feature_names_out()
    top_terms = [(feature_names[i], mean_scores[i]) for i in top_idx]
    print("\ntop-10 word features by mean tf-idf:")
    for term, score in top_terms:
        print(f"  {term!r}: {score:.4f}")


if __name__ == "__main__":
    main()
