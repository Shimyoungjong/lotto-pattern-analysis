"""
Playwright로 동행복권 allWinExel URL에서 전체 당첨번호 엑셀을 다운받는 모듈.
GitHub Actions 환경에서만 실행 (로컬 Playwright 설치 불필요).
"""

import pandas as pd
from pathlib import Path

BASE_DIR   = Path(__file__).parent.parent
EXCEL_PATH = BASE_DIR / "data" / "lotto_raw.xlsx"
EXCEL_URL  = "https://dhlottery.co.kr/gameResult.do?method=allWinExel"

# 엑셀 컬럼명 → 내부 표준 컬럼명 매핑
COLUMN_MAP = {
    "회차":       "round",
    "날짜":       "draw_date",
    "추첨일":     "draw_date",
    "1번":        "num1",
    "2번":        "num2",
    "3번":        "num3",
    "4번":        "num4",
    "5번":        "num5",
    "6번":        "num6",
    "번호1":      "num1",
    "번호2":      "num2",
    "번호3":      "num3",
    "번호4":      "num4",
    "번호5":      "num5",
    "번호6":      "num6",
    "보너스번호":  "bonus",
    "보너스":     "bonus",
}

REQUIRED_COLS = ["round", "draw_date", "num1", "num2", "num3", "num4", "num5", "num6", "bonus"]


def download_excel() -> Path:
    """
    Playwright 브라우저 세션을 통해 엑셀 파일을 다운받는다.
    - 메인 페이지 방문으로 WAF 통과 및 세션 쿠키 획득
    - context.request로 동일 세션에서 파일 직접 수신 (download 이벤트 불필요)
    """
    from playwright.sync_api import sync_playwright

    EXCEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page    = context.new_page()

        # WAF 챌린지 통과 — 메인 페이지에서 JS 실행 후 쿠키 획득
        print("메인 페이지 방문 중 (WAF 통과)...")
        page.goto("https://dhlottery.co.kr/", wait_until="networkidle")

        # 결과 페이지도 한번 방문해 세션 강화
        page.goto("https://dhlottery.co.kr/gameResult.do?method=allWin", wait_until="networkidle")

        # 브라우저 세션 쿠키를 그대로 사용해 파일 요청
        print(f"파일 요청 중: {EXCEL_URL}")
        response = context.request.get(
            EXCEL_URL,
            headers={"Referer": "https://dhlottery.co.kr/gameResult.do?method=allWin"},
        )

        if not response.ok:
            raise RuntimeError(f"다운로드 실패: HTTP {response.status}")

        content = response.body()

        # HTML이 반환된 경우 (WAF 미통과) 감지
        if content[:5] in (b"<html", b"\n\n\n\n\n", b"<!DOC"):
            raise RuntimeError("엑셀이 아닌 HTML 페이지가 반환됐습니다. WAF 통과 실패.")

        EXCEL_PATH.write_bytes(content)
        browser.close()

    size_kb = EXCEL_PATH.stat().st_size // 1024
    print(f"다운로드 완료: {EXCEL_PATH.name} ({size_kb} KB)")
    return EXCEL_PATH


def parse_excel(path: Path = EXCEL_PATH) -> pd.DataFrame:
    """엑셀을 읽어 표준 컬럼명으로 정규화된 DataFrame을 반환한다."""
    # .xlsx / .xls 순으로 엔진 시도
    df_raw = None
    for engine in ("openpyxl", "xlrd"):
        try:
            df_raw = pd.read_excel(path, engine=engine, header=None)
            break
        except Exception:
            continue

    if df_raw is None:
        raise ValueError(f"엑셀 파일을 읽을 수 없습니다: {path}")

    # '회차' 문자열이 있는 행을 헤더로 사용
    header_idx = _find_header_row(df_raw)
    df_raw.columns = df_raw.iloc[header_idx].astype(str).str.strip()
    df = df_raw.iloc[header_idx + 1:].reset_index(drop=True)

    # 표준 컬럼명으로 변환
    df = df.rename(columns=COLUMN_MAP)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"필수 컬럼 없음: {missing}\n"
            f"현재 컬럼: {df.columns.tolist()}"
        )

    df = df[REQUIRED_COLS].copy()

    # 타입 정규화
    df["round"]     = pd.to_numeric(df["round"], errors="coerce")
    df["draw_date"] = df["draw_date"].astype(str).str.strip()
    for col in ["num1", "num2", "num3", "num4", "num5", "num6", "bonus"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 결측·범위 이탈 행 제거
    df = df.dropna()
    num_cols = ["num1", "num2", "num3", "num4", "num5", "num6", "bonus"]
    mask = df[num_cols].apply(lambda col: col.between(1, 45)).all(axis=1)
    df   = df[mask].reset_index(drop=True)

    for col in ["round"] + num_cols:
        df[col] = df[col].astype(int)

    df = df.sort_values("round").reset_index(drop=True)
    print(f"파싱 완료: {len(df)}회차 ({df['round'].min()}~{df['round'].max()}회차)")
    return df


def _find_header_row(df: pd.DataFrame) -> int:
    """'회차' 문자열이 포함된 행 인덱스를 반환한다. 없으면 0."""
    for i, row in df.iterrows():
        if any("회차" in str(v) for v in row.values):
            return i
    return 0


def download_and_parse() -> pd.DataFrame:
    """다운로드 + 파싱을 순서대로 실행해 DataFrame을 반환한다."""
    path = download_excel()
    return parse_excel(path)
