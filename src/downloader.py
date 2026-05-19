"""
초기 데이터: data/lotto.xlsx 파일로 읽기
신규 데이터: 동행복권 API (getLottoNumber) 로 수집
"""

import time
import datetime
import requests
import pandas as pd
from pathlib import Path

BASE_DIR  = Path(__file__).parent.parent
XLSX_PATH = BASE_DIR / "data" / "lotto.xlsx"

API_URL = "https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo={}"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer":    "https://www.dhlottery.co.kr/",
}

COLUMN_MAP = {
    "회차": "round",      "날짜": "draw_date",    "추첨일": "draw_date",
    "1번":  "num1",       "2번":  "num2",          "3번":   "num3",
    "4번":  "num4",       "5번":  "num5",          "6번":   "num6",
    "번호1": "num1",      "번호2": "num2",         "번호3":  "num3",
    "번호4": "num4",      "번호5": "num5",         "번호6":  "num6",
    "보너스번호": "bonus", "보너스": "bonus",
}
REQUIRED_COLS = ["round", "draw_date", "num1", "num2", "num3", "num4", "num5", "num6", "bonus"]

_ROUND1_DATE = datetime.date(2002, 12, 7)

def _round_to_date(round_no: int) -> str:
    """회차 번호 → 추첨일 (1회차=2002-12-07, 매주 토요일)."""
    return (_ROUND1_DATE + datetime.timedelta(weeks=int(round_no) - 1)).strftime("%Y-%m-%d")


def load_xlsx(path: Path = XLSX_PATH) -> pd.DataFrame:
    """data/lotto.xlsx 를 읽어 표준 DataFrame으로 반환."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 파일이 없습니다.\n"
            "동행복권에서 전체 당첨번호 엑셀을 받아 data/lotto.xlsx 로 저장하세요."
        )

    df_raw = None
    for engine in ("openpyxl", "xlrd"):
        try:
            df_raw = pd.read_excel(path, engine=engine, header=None)
            break
        except Exception:
            continue
    if df_raw is None:
        raise ValueError(f"엑셀 파일을 읽을 수 없습니다: {path}")

    # '회차' 문자열이 있는 행을 헤더로 찾기
    header_idx = 0
    for i, row in df_raw.iterrows():
        if any("회차" in str(v) for v in row.values):
            header_idx = i
            break

    df_raw.columns = df_raw.iloc[header_idx].astype(str).str.strip()
    df = df_raw.iloc[header_idx + 1:].reset_index(drop=True)
    df = df.rename(columns=COLUMN_MAP)

    # draw_date 없으면 회차 번호로 계산
    if "draw_date" not in df.columns and "round" in df.columns:
        df["draw_date"] = pd.to_numeric(df["round"], errors="coerce").apply(
            lambda r: _round_to_date(r) if pd.notna(r) else ""
        )

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼 없음: {missing}\n현재 컬럼: {df.columns.tolist()}")

    df = df[REQUIRED_COLS].copy()
    df["round"]     = pd.to_numeric(df["round"],     errors="coerce")
    df["draw_date"] = df["draw_date"].astype(str).str.strip()
    for col in ["num1", "num2", "num3", "num4", "num5", "num6", "bonus"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()
    num_cols = ["num1", "num2", "num3", "num4", "num5", "num6", "bonus"]
    mask = df[num_cols].apply(lambda c: c.between(1, 45)).all(axis=1)
    df = df[mask].reset_index(drop=True)
    for col in ["round"] + num_cols:
        df[col] = df[col].astype(int)

    df = df.sort_values("round").reset_index(drop=True)
    print(f"xlsx 로드 완료: {len(df)}회차 ({df['round'].min()}~{df['round'].max()}회차)")
    return df


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
    print(f"최신 회차 (API): {lo}")
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


def fetch_new_rounds(start: int, end: int) -> pd.DataFrame:
    """start~end 회차를 API로 수집해 DataFrame으로 반환."""
    rows = []
    for drw_no in range(start, end + 1):
        row = _fetch_one(drw_no)
        if row:
            rows.append(row)
        time.sleep(0.15)
        if drw_no % 50 == 0:
            print(f"수집 중: {drw_no}/{end}회차")

    if not rows:
        print(f"신규 수집 데이터 없음 ({start}~{end}회차)")
        return pd.DataFrame(columns=REQUIRED_COLS)

    df = pd.DataFrame(rows)
    df = df.sort_values("round").reset_index(drop=True)
    print(f"API 수집 완료: {len(df)}회차 ({df['round'].min()}~{df['round'].max()}회차)")
    return df
