"""
SQLite DB로 로또 데이터를 저장·조회하는 모듈.
"""

import sqlite3
import pandas as pd
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).parent.parent
DB_PATH  = BASE_DIR / "data" / "lotto.db"

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS lotto_results (
    round     INTEGER PRIMARY KEY,
    draw_date TEXT NOT NULL,
    num1      INTEGER NOT NULL,
    num2      INTEGER NOT NULL,
    num3      INTEGER NOT NULL,
    num4      INTEGER NOT NULL,
    num5      INTEGER NOT NULL,
    num6      INTEGER NOT NULL,
    bonus     INTEGER NOT NULL
)
"""


def get_connection() -> sqlite3.Connection:
    """DB 커넥션을 반환하고, 테이블이 없으면 생성한다."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(CREATE_TABLE_SQL)
    conn.commit()
    return conn


def save_to_db(df: pd.DataFrame) -> int:
    """
    DataFrame 전체를 DB에 저장한다. 기존 데이터는 모두 교체(REPLACE).
    새 CSV로 전체 갱신할 때 사용.
    """
    if df.empty:
        return 0

    conn = get_connection()
    df.to_sql("lotto_results", conn, if_exists="replace", index=False,
              method="multi", chunksize=500)
    conn.commit()
    conn.close()
    print(f"DB 저장 완료: {len(df)}회차")
    return len(df)


def load_from_db() -> pd.DataFrame:
    """DB에서 전체 데이터를 DataFrame으로 불러온다."""
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM lotto_results ORDER BY round", conn)
    conn.close()
    return df


def get_latest_round() -> int:
    """DB에 저장된 가장 최신 회차 번호를 반환한다. 없으면 0 반환."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(round) FROM lotto_results")
    result = cursor.fetchone()[0]
    conn.close()
    return result if result is not None else 0


def get_db_info() -> dict:
    """DB 상태 요약 정보를 반환한다."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*), MIN(round), MAX(round) FROM lotto_results")
    count, min_r, max_r = cursor.fetchone()
    conn.close()
    return {"total": count or 0, "min_round": min_r or 0, "max_round": max_r or 0}
