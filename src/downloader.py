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
    Playwright 브라우저 내부의 JS fetch()로 엑셀을 다운받는다.
    context.request / page.goto 방식은 WAF 재검사에 걸리므로,
    이미 WAF를 통과한 브라우저 페이지 컨텍스트에서 직접 fetch() 실행.
    """
    from playwright.sync_api import sync_playwright

    EXCEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
            ],
        )
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            accept_downloads=True,
            locale="ko-KR",
            extra_http_headers={"Accept-Language": "ko-KR,ko;q=0.9"},
        )
        # navigator.webdriver 속성 제거 (자동화 감지 우회)
        context.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )
        page = context.new_page()

        # 메인 → 결과 페이지 순서로 방문해 WAF 세션 완전히 수립
        print("메인 페이지 방문 중 (WAF 통과)...")
        page.goto("https://dhlottery.co.kr/", wait_until="domcontentloaded", timeout=90_000)
        page.wait_for_timeout(3_000)

        print("결과 페이지 방문 중...")
        page.goto("https://dhlottery.co.kr/gameResult.do?method=allWin",
                  wait_until="domcontentloaded", timeout=90_000)
        page.wait_for_timeout(2_000)
        print(f"현재 URL : {page.url}")
        print(f"현재 타이틀: {page.title()}")

        # 엑셀 다운로드 버튼/링크 탐색
        selectors = [
            "a[href*='allWinExel']",
            "a:has-text('엑셀')",
            "input[value*='엑셀']",
            "button:has-text('엑셀')",
            "a:has-text('다운로드')",
        ]
        btn = None
        for sel in selectors:
            el = page.locator(sel)
            if el.count() > 0:
                print(f"버튼 발견: {sel}")
                btn = el.first
                break

        print("다운로드 시작...")
        with page.expect_download(timeout=120_000) as dl:
            if btn:
                btn.click()
            else:
                # wait_until="commit" — 파일 응답이 시작되면 바로 반환
                print("버튼 없음 → page.goto commit 방식으로 시도...")
                page.goto(EXCEL_URL, wait_until="commit", timeout=120_000)

        download = dl.value
        download.save_as(str(EXCEL_PATH))
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
