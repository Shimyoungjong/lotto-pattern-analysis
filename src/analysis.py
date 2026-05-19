"""
로또 당첨번호 통계 분석 및 시각화 모듈.
- 번호별 출현 빈도
- 홀짝 비율
- 고저 비율 (1~22 저번호, 23~45 고번호)
- 연속번호 패턴
"""

import matplotlib
matplotlib.use("Agg")  # 헤드리스 환경(GitHub Actions) 대응
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from pathlib import Path
from typing import Tuple

# 한글 폰트: Windows=맑은고딕, Linux=나눔고딕 또는 기본 폰트
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

# 번호 컬럼 (보너스 제외)
NUM_COLS = ["num1", "num2", "num3", "num4", "num5", "num6"]


def _get_all_numbers(df: pd.DataFrame) -> pd.Series:
    """DataFrame에서 당첨번호 전체를 1차원 Series로 반환한다."""
    return df[NUM_COLS].values.flatten()


# ──────────────────────────────────────────────
# 1. 번호별 출현 빈도
# ──────────────────────────────────────────────

def frequency_analysis(df: pd.DataFrame) -> pd.Series:
    """1~45 각 번호의 누적 출현 횟수를 반환한다."""
    all_nums = _get_all_numbers(df)
    freq = pd.Series(all_nums).value_counts().sort_index()
    # 한 번도 안 나온 번호도 0으로 채움
    freq = freq.reindex(range(1, 46), fill_value=0)
    return freq


def plot_frequency(df: pd.DataFrame) -> None:
    """번호별 출현 빈도 막대 그래프를 저장한다."""
    freq = frequency_analysis(df)

    fig, ax = plt.subplots(figsize=(16, 6))
    colors = ["#E74C3C" if v == freq.max() else "#3498DB" for v in freq.values]
    ax.bar(freq.index, freq.values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_title(f"로또 번호별 출현 빈도 (총 {len(df)}회차)", fontsize=16, fontweight="bold")
    ax.set_xlabel("번호")
    ax.set_ylabel("출현 횟수")
    ax.set_xticks(range(1, 46))
    ax.tick_params(axis="x", labelsize=8)

    # 가장 많이 나온 번호 표시
    max_num = freq.idxmax()
    ax.annotate(f"최다: {max_num}번\n({freq.max()}회)",
                xy=(max_num, freq.max()), xytext=(max_num + 2, freq.max() + 1),
                arrowprops=dict(arrowstyle="->", color="black"), fontsize=9)

    plt.tight_layout()
    path = FIGURES_DIR / "frequency.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"저장: {path}")


# ──────────────────────────────────────────────
# 2. 홀짝 비율
# ──────────────────────────────────────────────

def odd_even_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """회차별 홀수/짝수 개수를 계산해 반환한다."""
    result = df[["round"]].copy()
    nums = df[NUM_COLS]
    result["odd_count"]  = (nums % 2 == 1).sum(axis=1)
    result["even_count"] = (nums % 2 == 0).sum(axis=1)
    return result


def plot_odd_even(df: pd.DataFrame) -> None:
    """홀짝 비율 분포를 시각화한다."""
    oe = odd_even_analysis(df)
    odd_dist = oe["odd_count"].value_counts().sort_index()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 홀수 개수 분포
    axes[0].bar(odd_dist.index, odd_dist.values, color="#9B59B6", edgecolor="white")
    axes[0].set_title("회차별 홀수 개수 분포", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("홀수 개수 (6개 중)")
    axes[0].set_ylabel("회차 수")
    axes[0].set_xticks(range(0, 7))

    # 전체 홀짝 비율 파이차트
    total_odd  = (df[NUM_COLS] % 2 == 1).values.sum()
    total_even = (df[NUM_COLS] % 2 == 0).values.sum()
    axes[1].pie([total_odd, total_even],
                labels=["홀수", "짝수"],
                autopct="%1.1f%%",
                colors=["#9B59B6", "#F39C12"],
                startangle=90)
    axes[1].set_title("전체 홀짝 비율", fontsize=13, fontweight="bold")

    plt.tight_layout()
    path = FIGURES_DIR / "odd_even.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"저장: {path}")


# ──────────────────────────────────────────────
# 3. 고저 비율 (1~22: 저번호, 23~45: 고번호)
# ──────────────────────────────────────────────

def high_low_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """회차별 저번호(1~22)/고번호(23~45) 개수를 계산해 반환한다."""
    result = df[["round"]].copy()
    nums = df[NUM_COLS]
    result["low_count"]  = (nums <= 22).sum(axis=1)
    result["high_count"] = (nums >= 23).sum(axis=1)
    return result


def plot_high_low(df: pd.DataFrame) -> None:
    """고저 비율 분포를 시각화한다."""
    hl = high_low_analysis(df)
    low_dist = hl["low_count"].value_counts().sort_index()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(low_dist.index, low_dist.values, color="#27AE60", edgecolor="white")
    axes[0].set_title("회차별 저번호(1~22) 개수 분포", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("저번호 개수 (6개 중)")
    axes[0].set_ylabel("회차 수")
    axes[0].set_xticks(range(0, 7))

    total_low  = (df[NUM_COLS] <= 22).values.sum()
    total_high = (df[NUM_COLS] >= 23).values.sum()
    axes[1].pie([total_low, total_high],
                labels=["저번호 (1~22)", "고번호 (23~45)"],
                autopct="%1.1f%%",
                colors=["#27AE60", "#E74C3C"],
                startangle=90)
    axes[1].set_title("전체 고저 비율", fontsize=13, fontweight="bold")

    plt.tight_layout()
    path = FIGURES_DIR / "high_low.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"저장: {path}")


# ──────────────────────────────────────────────
# 4. 연속번호 패턴
# ──────────────────────────────────────────────

def consecutive_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    회차별 연속번호(차이가 1인 번호 쌍) 개수를 계산한다.
    예: [3, 4, 7, 8, 9, 15] → 연속쌍 2개
    """
    def count_consecutive(row):
        nums = sorted(row)
        count = sum(1 for a, b in zip(nums, nums[1:]) if b - a == 1)
        return count

    result = df[["round"]].copy()
    result["consecutive_pairs"] = df[NUM_COLS].apply(count_consecutive, axis=1)
    return result


def plot_consecutive(df: pd.DataFrame) -> None:
    """연속번호 패턴 분포를 시각화한다."""
    consec = consecutive_analysis(df)
    dist = consec["consecutive_pairs"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(dist.index, dist.values, color="#1ABC9C", edgecolor="white")
    ax.set_title("회차별 연속번호 쌍 개수 분포", fontsize=13, fontweight="bold")
    ax.set_xlabel("연속번호 쌍 개수")
    ax.set_ylabel("회차 수")
    ax.set_xticks(dist.index)

    for i, v in zip(dist.index, dist.values):
        ax.text(i, v + 1, f"{v}회\n({v/len(df)*100:.1f}%)", ha="center", fontsize=9)

    plt.tight_layout()
    path = FIGURES_DIR / "consecutive.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"저장: {path}")


# ──────────────────────────────────────────────
# 전체 분석 실행
# ──────────────────────────────────────────────

def run_all_analysis(df: pd.DataFrame) -> dict:
    """모든 분석을 실행하고 결과 요약 딕셔너리를 반환한다."""
    print("\n[분석 시작]")

    freq  = frequency_analysis(df)
    oe    = odd_even_analysis(df)
    hl    = high_low_analysis(df)
    consec = consecutive_analysis(df)

    print("  번호별 출현 빈도 시각화 중...")
    plot_frequency(df)
    print("  홀짝 비율 시각화 중...")
    plot_odd_even(df)
    print("  고저 비율 시각화 중...")
    plot_high_low(df)
    print("  연속번호 패턴 시각화 중...")
    plot_consecutive(df)

    summary = {
        "total_rounds":        len(df),
        "most_frequent_num":   int(freq.idxmax()),
        "least_frequent_num":  int(freq.idxmin()),
        "avg_odd_per_round":   round(oe["odd_count"].mean(), 2),
        "avg_low_per_round":   round(hl["low_count"].mean(), 2),
        "consec_pair_rate":    round((consec["consecutive_pairs"] > 0).mean() * 100, 1),
    }

    print("\n[분석 요약]")
    print(f"  총 회차: {summary['total_rounds']}회")
    print(f"  최다 출현 번호: {summary['most_frequent_num']}번")
    print(f"  최소 출현 번호: {summary['least_frequent_num']}번")
    print(f"  회차당 평균 홀수 개수: {summary['avg_odd_per_round']}개")
    print(f"  회차당 평균 저번호 개수: {summary['avg_low_per_round']}개")
    print(f"  연속번호 포함 회차 비율: {summary['consec_pair_rate']}%")

    return summary
