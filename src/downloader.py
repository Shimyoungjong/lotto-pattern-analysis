"""
동행복권 공식 API로 당첨번호를 수집하는 모듈.
URL: https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo={회차}
"""

import time
import requests
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
API_URL  = "https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo={}"
HEADERS  = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer":    "https://www.dhlottery.co.kr/",
}


def fetch_latest_round() -> int:
    """이진 탐색으로 현재 최신 회차를 찾는다."""
    lo, hi = 1, 2000
    while lo < hi:
        mid = (lo + hi + 1) // 2
        try:
            data = requests.get(API_URL.format(mid), headers=HEADERS, timeout=10).json()
            if data.get("returnValue") == "success":
                lo = mid
            else:
                hi = mid - 1
        except Exception:
            hi = mid - 1
    print(f"최신 회차: {lo}")
    return lo


def _fetch_one(drw_no: int) -> dict | None:
    for attempt in range(3):
        try:
            data = requests.get(API_URL.format(drw_no), headers=HEADERS, timeout=10).json()
            if data.get("returnValue") == "success":
                return {
                    "round":     int(data["drwNo"]),
                    "draw_date": str(data["drwNoDate"]),
                    "num1":      int(data["drwtNo1"]),
                    "num2":      int(data["drwtNo2"]),
                    "num3":      int(data["drwtNo3"]),
                    "num4":      int(data["drwtNo4"]),
                    "num5":      int(data["drwtNo5"]),
                    "num6":      int(data["drwtNo6"]),
                    "bonus":     int(data["bnusNo"]),
                }
            return None
        except Exception as e:
            if attempt == 2:
                print(f"  {drw_no}회차 실패: {e}")
            time.sleep(1)
    return None


def fetch_rounds(start: int, end: int) -> pd.DataFrame:
    """start~end 회차를 API로 수집해 DataFrame으로 반환."""
    rows = []
    for drw_no in range(start, end + 1):
        row = _fetch_one(drw_no)
        if row:
            rows.append(row)
        time.sleep(0.15)
        if drw_no % 100 == 0:
            print(f"수집 중: {drw_no}/{end}회차")

    if not rows:
        raise ValueError(f"{start}~{end} 회차 수집 결과 없음")

    df = pd.DataFrame(rows)
    df = df.sort_values("round").reset_index(drop=True)
    print(f"수집 완료: {len(df)}회차 ({df['round'].min()}~{df['round'].max()}회차)")
    return df


def download_and_parse(start: int = 1, end: int = None) -> pd.DataFrame:
    """전체 또는 지정 범위 회차를 수집해 DataFrame으로 반환."""
    if end is None:
        end = fetch_latest_round()
    return fetch_rounds(start, end)
