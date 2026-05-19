"""
로또 번호 패턴 분석을 위한 머신러닝 모델 모듈.
- Random Forest: 번호별 출현 여부 분류
- K-Means: 당첨 번호 조합 클러스터링
- Apriori: 자주 함께 등장하는 번호 조합 연관규칙 분석
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
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

def _set_korean_font():
    candidates = ["Malgun Gothic", "NanumGothic", "NanumBarunGothic", "DejaVu Sans"]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            plt.rcParams["font.family"] = font
            break
    plt.rcParams["axes.unicode_minus"] = False

_set_korean_font()

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


# ──────────────────────────────────────────────
# 번호 추천
# ──────────────────────────────────────────────

def _normalize(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-9)


def _top6(scores: np.ndarray) -> list[int]:
    """scores[1..45] 기준 상위 6개 번호를 정렬해서 반환."""
    return sorted((np.argsort(scores[1:])[::-1][:6] + 1).tolist())


def _rf_scores(df: pd.DataFrame) -> np.ndarray:
    """각 번호가 다음 회차에 등장할 RF 확률 (1-indexed, 인덱스 0 미사용)."""
    from sklearn.multioutput import MultiOutputClassifier

    print("  [RF] 학습 중 (약 30초)...")
    one_hot = _to_one_hot(df)
    WINDOW  = 10

    X, Y = [], []
    for i in range(WINDOW, len(one_hot)):
        X.append(one_hot.iloc[i - WINDOW:i].values.flatten())
        Y.append(one_hot.iloc[i].values)
    X, Y = np.array(X), np.array(Y)

    clf = MultiOutputClassifier(
        RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
        n_jobs=-1,
    )
    clf.fit(X, Y)

    X_latest = one_hot.iloc[-WINDOW:].values.flatten().reshape(1, -1)
    proba = np.array([
        est.predict_proba(X_latest)[0][1] if len(est.classes_) > 1 else 0.0
        for est in clf.estimators_
    ])

    scores = np.zeros(46)
    scores[1:] = proba
    return _normalize(scores)


def _kmeans_scores(df: pd.DataFrame) -> np.ndarray:
    """최근 20회차가 주로 속한 클러스터의 번호 등장 빈도."""
    print("  [K-Means] 클러스터 분석 중...")
    one_hot = _to_one_hot(df)
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(one_hot)

    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    recent_cluster = pd.Series(labels[-20:]).value_counts().index[0]
    cluster_data   = df[NUM_COLS][labels == recent_cluster]

    scores = np.zeros(46)
    for num in range(1, 46):
        scores[num] = (cluster_data.values == num).sum()
    return _normalize(scores)


def _apriori_scores(df: pd.DataFrame) -> np.ndarray:
    """Apriori 빈발 아이템셋에서 각 번호의 누적 support."""
    print("  [Apriori] 연관규칙 분석 중...")
    transactions = df[NUM_COLS].apply(
        lambda r: [str(n) for n in sorted(r)], axis=1
    ).tolist()

    te     = TransactionEncoder()
    te_df  = pd.DataFrame(te.fit_transform(transactions), columns=te.columns_)
    freq   = apriori(te_df, min_support=0.01, use_colnames=True)

    scores = np.zeros(46)
    for _, row in freq.iterrows():
        for item in row["itemsets"]:
            scores[int(item)] += row["support"]
    return _normalize(scores)


def _plot_recommendation(rf: np.ndarray, km: np.ndarray,
                         ap: np.ndarray, combined: np.ndarray,
                         final_6: list[int]) -> None:
    nums = np.arange(1, 46)

    rf_top6 = _top6(rf)
    km_top6 = _top6(km)
    ap_top6 = _top6(ap)

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    fig.suptitle("로또 번호 추천 분석", fontsize=16, fontweight="bold")

    configs = [
        (rf[1:],       "Random Forest 확률",   "#3498DB", rf_top6),
        (km[1:],       "K-Means 클러스터",     "#E74C3C", km_top6),
        (ap[1:],       "Apriori 연관규칙",     "#2ECC71", ap_top6),
        (combined[1:], "★ 최종 통합 추천",     "#9B59B6", final_6),
    ]
    for ax, (scores, title, color, top6) in zip(axes.flat, configs):
        bar_colors = [color if n in top6 else "#D5D8DC" for n in nums]
        ax.bar(nums, scores, color=bar_colors, edgecolor="white", linewidth=0.4)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("번호")
        ax.set_ylabel("점수")
        ax.set_xticks(range(1, 46))
        ax.tick_params(axis="x", labelsize=7)
        for n in top6:
            ax.text(n, scores[n - 1] + 0.01, str(n),
                    ha="center", va="bottom", fontsize=7, fontweight="bold", color=color)

    plt.tight_layout()
    path = FIGURES_DIR / "recommendation.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  저장: {path}")


def recommend_numbers(df: pd.DataFrame) -> dict:
    """RF + K-Means + Apriori 결합으로 당첨 예상 번호 6개 추천."""
    print("\n========== 번호 추천 시작 ==========")

    rf  = _rf_scores(df)
    km  = _kmeans_scores(df)
    ap  = _apriori_scores(df)

    # 가중 합산: RF 40% + KMeans 30% + Apriori 30%
    combined = _normalize(rf * 0.4 + km * 0.3 + ap * 0.3)

    rf_top6  = _top6(rf)
    km_top6  = _top6(km)
    ap_top6  = _top6(ap)
    final_6  = _top6(combined)

    print(f"\n  [Random Forest]  {rf_top6}")
    print(f"  [K-Means]        {km_top6}")
    print(f"  [Apriori]        {ap_top6}")
    print(f"\n  ★ 최종 추천 번호: {final_6}")

    _plot_recommendation(rf, km, ap, combined, final_6)
    print("\n========== 번호 추천 완료 ==========")

    return {"random_forest": rf_top6, "kmeans": km_top6,
            "apriori": ap_top6, "final": final_6}
