"""
로또 번호 패턴 분석을 위한 머신러닝 모델 모듈.
- Random Forest: 번호별 출현 여부 분류
- K-Means: 당첨 번호 조합 클러스터링
- Apriori: 자주 함께 등장하는 번호 조합 연관규칙 분석
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR    = Path(__file__).parent.parent
FIGURES_DIR = BASE_DIR / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

NUM_COLS = ["num1", "num2", "num3", "num4", "num5", "num6"]


# ──────────────────────────────────────────────
# 공통 유틸
# ──────────────────────────────────────────────

def _to_one_hot(df: pd.DataFrame) -> pd.DataFrame:
    """
    각 회차의 번호를 45차원 원-핫 인코딩으로 변환한다.
    컬럼명: num_1 ~ num_45
    """
    matrix = np.zeros((len(df), 45), dtype=int)
    for i, row in enumerate(df[NUM_COLS].values):
        for num in row:
            matrix[i, num - 1] = 1
    cols = [f"num_{i}" for i in range(1, 46)]
    return pd.DataFrame(matrix, columns=cols)


# ──────────────────────────────────────────────
# 1. Random Forest — 번호 출현 분류
# ──────────────────────────────────────────────

def run_random_forest(df: pd.DataFrame) -> dict:
    """
    각 번호(1~45)가 다음 회차에 출현하는지 예측하는 Random Forest 모델.
    특징(feature): 최근 N회차의 번호별 출현 이력 (슬라이딩 윈도우)
    레이블(label): 해당 회차에 해당 번호 출현 여부

    단순 예측보다 특징 중요도(feature importance) 파악에 의미 있음.
    """
    print("\n[Random Forest] 학습 시작...")

    one_hot = _to_one_hot(df)
    WINDOW  = 10  # 직전 10회차 이력을 특징으로 사용

    results = {}

    # 대표로 번호 7을 예시 타깃으로 모델 학습 (전체 45개 번호에 동일 적용 가능)
    target_col = "num_7"

    X_list, y_list = [], []
    for i in range(WINDOW, len(one_hot)):
        X_list.append(one_hot.iloc[i - WINDOW:i].values.flatten())
        y_list.append(one_hot.iloc[i][target_col])

    X = np.array(X_list)
    y = np.array(y_list)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    print(f"  정확도: {report['accuracy']:.3f}")

    # 특징 중요도 시각화 (상위 20개 회차-번호 조합)
    importances = clf.feature_importances_
    top_idx = np.argsort(importances)[-20:][::-1]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(20), importances[top_idx], color="#3498DB")
    ax.set_title(f"Random Forest 특징 중요도 (타깃: {target_col})", fontsize=13, fontweight="bold")
    ax.set_xlabel("특징 인덱스 (상위 20개)")
    ax.set_ylabel("중요도")
    plt.tight_layout()
    path = FIGURES_DIR / "rf_importance.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  저장: {path}")

    results["accuracy"] = report["accuracy"]
    results["model"]    = clf
    return results


# ──────────────────────────────────────────────
# 2. K-Means — 번호 조합 클러스터링
# ──────────────────────────────────────────────

def run_kmeans(df: pd.DataFrame, n_clusters: int = 5) -> pd.DataFrame:
    """
    당첨 번호 조합을 K-Means로 클러스터링한다.
    원-핫 인코딩 후 PCA로 2차원 축소, 시각화.
    """
    print(f"\n[K-Means] {n_clusters}개 클러스터 분석 시작...")

    one_hot = _to_one_hot(df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(one_hot)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    # PCA로 2D 시각화
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(10, 7))
    palette = sns.color_palette("tab10", n_clusters)
    for k in range(n_clusters):
        mask = labels == k
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   label=f"클러스터 {k + 1} ({mask.sum()}회)", alpha=0.6, s=20, color=palette[k])

    ax.set_title(f"K-Means 클러스터링 (PCA 2D, k={n_clusters})", fontsize=13, fontweight="bold")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    path = FIGURES_DIR / "kmeans_cluster.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  저장: {path}")

    # 클러스터별 대표 번호 출력
    result_df = df.copy()
    result_df["cluster"] = labels
    print("\n  클러스터별 평균 번호:")
    for k in range(n_clusters):
        subset = df[NUM_COLS][labels == k]
        avg = subset.values.flatten()
        top_nums = pd.Series(avg).value_counts().head(5).index.tolist()
        print(f"    클러스터 {k + 1}: 자주 등장 번호 {top_nums}")

    return result_df


# ──────────────────────────────────────────────
# 3. Apriori — 연관규칙 분석
# ──────────────────────────────────────────────

def run_apriori(df: pd.DataFrame, min_support: float = 0.05, min_confidence: float = 0.3) -> pd.DataFrame:
    """
    Apriori 알고리즘으로 자주 함께 등장하는 번호 조합의 연관규칙을 분석한다.
    min_support: 전체 회차 중 해당 조합이 등장하는 최소 비율
    min_confidence: 규칙의 신뢰도 최솟값
    """
    print(f"\n[Apriori] 연관규칙 분석 시작 (min_support={min_support}, min_confidence={min_confidence})...")

    # 각 회차를 번호 집합의 트랜잭션으로 변환
    transactions = df[NUM_COLS].apply(lambda row: [str(n) for n in sorted(row)], axis=1).tolist()

    te = TransactionEncoder()
    te_array = te.fit_transform(transactions)
    te_df = pd.DataFrame(te_array, columns=te.columns_)

    # 빈번 아이템셋 추출
    frequent_itemsets = apriori(te_df, min_support=min_support, use_colnames=True)
    if frequent_itemsets.empty:
        print("  빈번 아이템셋 없음. min_support를 낮춰보세요.")
        return pd.DataFrame()

    frequent_itemsets["length"] = frequent_itemsets["itemsets"].apply(len)
    print(f"  빈번 아이템셋 수: {len(frequent_itemsets)}개")

    # 연관규칙 생성
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    rules = rules.sort_values("lift", ascending=False)
    print(f"  연관규칙 수: {len(rules)}개")

    # 상위 규칙 출력
    print("\n  [상위 10개 연관규칙 (lift 기준)]")
    for _, row in rules.head(10).iterrows():
        ant = sorted(list(row["antecedents"]))
        con = sorted(list(row["consequents"]))
        print(f"    {ant} → {con}  |  support={row['support']:.3f}, "
              f"confidence={row['confidence']:.3f}, lift={row['lift']:.3f}")

    # 2-번호 조합 히트맵
    _plot_pair_heatmap(df)

    return rules


def _plot_pair_heatmap(df: pd.DataFrame) -> None:
    """번호 쌍의 동시 등장 빈도를 히트맵으로 시각화한다."""
    matrix = np.zeros((45, 45), dtype=int)
    for _, row in df[NUM_COLS].iterrows():
        nums = sorted(row.tolist())
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                a, b = nums[i] - 1, nums[j] - 1
                matrix[a][b] += 1
                matrix[b][a] += 1

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(matrix, cmap="YlOrRd", ax=ax,
                xticklabels=range(1, 46), yticklabels=range(1, 46),
                linewidths=0, cbar_kws={"label": "동시 등장 횟수"})
    ax.set_title("번호 쌍 동시 등장 빈도 히트맵", fontsize=14, fontweight="bold")
    ax.tick_params(axis="both", labelsize=7)
    plt.tight_layout()
    path = FIGURES_DIR / "pair_heatmap.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  저장: {path}")


# ──────────────────────────────────────────────
# 전체 모델 실행
# ──────────────────────────────────────────────

def run_all_models(df: pd.DataFrame) -> None:
    """모든 ML 모델을 순서대로 실행한다."""
    print("\n========== ML 모델 분석 시작 ==========")
    run_random_forest(df)
    run_kmeans(df)
    run_apriori(df)
    print("\n========== ML 모델 분석 완료 ==========")
