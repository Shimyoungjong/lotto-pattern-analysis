"""
CSV 파일을 읽어 정규화된 DataFrame으로 변환하는 모듈.

지원하는 CSV 컬럼 형식:
  1. 표준형:   round, draw_date, num1~6, bonus
  2. 한글형:   회차, 날짜, 번호1~6, 보너스번호
  3. API형:    drwNo, drwNoDate, drwtNo1~6, bnusNo

data/lotto_input.csv 에 파일을 두고 실행하면 됩니다.
매주 새 CSV로 교체 후 --update 를 실행하면 DB가 갱신됩니다.
"""

import pandas as pd
from pathlib import Path

BASE_DIR   = Path(__file__).parent.parent
INPUT_CSV  = BASE_DIR / "data" / "lotto_input.csv"

# 외부 CSV의 다양한 컬럼명 → 내부 표준 컬럼명 매핑
COLUMN_MAP = {
    # 한글형
    "회차":      "round",
    "날짜":      "draw_date",
    "추첨일":    "draw_date",
    "번호1":     "num1",
    "번호2":     "num2",
    "번호3":     "num3",
    "번호4":     "num4",
    "번호5":     "num5",
    "번호6":     "num6",
    "보너스번호": "bonus",
    "보너스":    "bonus",
    # API형
    "drwNo":     "round",
    "drwNoDate": "draw_date",
    "drwtNo1":   "num1",
    "drwtNo2":   "num2",
    "drwtNo3":   "num3",
    "drwtNo4":   "num4",
    "drwtNo5":   "num5",
    "drwtNo6":   "num6",
    "bnusNo":    "bonus",
}

REQUIRED_COLS = ["round", "draw_date", "num1", "num2", "num3", "num4", "num5", "num6", "bonus"]


def load_csv(path: Path = INPUT_CSV) -> pd.DataFrame:
    """
    CSV 파일을 읽어 표준 컬럼명으로 정규화된 DataFrame을 반환한다.
    파일이 없거나 필수 컬럼이 부족하면 예외를 발생시킨다.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"CSV 파일을 찾을 수 없습니다: {path}\n"
            f"로또 당첨번호 CSV를 다운받아 해당 경로에 저장해 주세요."
        )

    # 인코딩 자동 감지 (UTF-8 → CP949 순으로 시도)
    df = _read_with_encoding(path)

    # 컬럼명 공백 제거 후 표준명으로 변환
    df.columns = df.columns.str.strip()
    df = df.rename(columns=COLUMN_MAP)

    # 필수 컬럼 존재 확인
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV에 필수 컬럼이 없습니다: {missing}\n"
            f"현재 컬럼: {df.columns.tolist()}"
        )

    df = df[REQUIRED_COLS].copy()

    # 타입 정규화
    df["round"]     = pd.to_numeric(df["round"], errors="coerce").astype("Int64")
    df["draw_date"] = df["draw_date"].astype(str).str.strip()
    for col in ["num1", "num2", "num3", "num4", "num5", "num6", "bonus"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # 결측값·범위 이탈 행 제거
    df = df.dropna()
    num_cols = ["num1", "num2", "num3", "num4", "num5", "num6", "bonus"]
    mask = df[num_cols].apply(lambda col: col.between(1, 45)).all(axis=1)
    df  = df[mask].reset_index(drop=True)

    # pandas Int64 → 일반 int (SQLite 호환)
    for col in ["round"] + num_cols:
        df[col] = df[col].astype(int)

    df = df.sort_values("round").reset_index(drop=True)
    print(f"CSV 로드 완료: {len(df)}회차 ({path.name})")
    return df


def _read_with_encoding(path: Path) -> pd.DataFrame:
    """UTF-8-sig → UTF-8 → CP949 순으로 인코딩을 시도해 CSV를 읽는다."""
    for enc in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return pd.read_csv(path, encoding=enc)
        except (UnicodeDecodeError, Exception):
            continue
    raise ValueError(f"CSV 파일 인코딩을 인식할 수 없습니다: {path}")
