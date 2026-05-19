"""
로또 번호 분석 프로젝트 진입점.

사용법:
  python main.py --download   xlsx 초기 로드 + API 신규 회차 추가
  python main.py --info       DB 현재 상태 출력
  python main.py --analyze    통계 분석 + 시각화
  python main.py --train      ML 모델 학습
  python main.py --all        download → analyze → train 전체 파이프라인
"""

import argparse
import sys


def run_download():
    """xlsx로 초기 DB 구축, 이후 신규 회차는 API로 추가."""
    from src.downloader import load_xlsx, fetch_new_rounds, fetch_latest_round, XLSX_PATH
    from src.database import upsert_to_db, get_latest_round

    print("=== 데이터 업데이트 ===")
    db_latest = get_latest_round()

    # DB가 비어있으면 xlsx에서 초기 로드
    if db_latest == 0:
        print("DB 비어있음 → data/lotto.xlsx 초기 로드 시작...")
        df_xlsx = load_xlsx(XLSX_PATH)
        upsert_to_db(df_xlsx)
        db_latest = get_latest_round()
        print(f"초기 로드 완료: {db_latest}회차까지 DB 저장")

    # 신규 회차 API 수집
    try:
        api_latest = fetch_latest_round()
    except Exception as e:
        print(f"최신 회차 조회 실패: {e}")
        return

    if api_latest <= db_latest:
        if api_latest < db_latest:
            print(
                f"[경고] API 최신 회차({api_latest})가 DB({db_latest})보다 낮음.\n"
                "       동행복권 WAF가 API를 차단하고 있습니다.\n"
                "       신규 회차는 data/lotto.xlsx 파일을 교체 후 --download 를 다시 실행하세요."
            )
        else:
            print(f"이미 최신 상태 ({db_latest}회차). 업데이트 불필요.")
        return

    start = db_latest + 1
    print(f"신규 회차 수집: {start}~{api_latest}회차...")
    new_df = fetch_new_rounds(start, api_latest)
    if not new_df.empty:
        upsert_to_db(new_df)
    else:
        print("API 수집 결과 없음 (WAF 차단 또는 아직 미발표).")


def run_info():
    from src.database import get_db_info

    info = get_db_info()
    if info["total"] == 0:
        print("DB가 비어 있습니다. --download 를 먼저 실행하세요.")
    else:
        print(f"DB 상태: 총 {info['total']}회차  |  {info['min_round']}~{info['max_round']}회차")


def run_analyze():
    from src.database import load_from_db
    from src.analysis import run_all_analysis

    print("=== 통계 분석 시작 ===")
    df = load_from_db()
    if df.empty:
        print("데이터가 없습니다. --download 를 먼저 실행하세요.")
        sys.exit(1)
    run_all_analysis(df)


def run_train():
    from src.database import load_from_db
    from src.models import run_all_models

    print("=== ML 모델 학습 시작 ===")
    df = load_from_db()
    if df.empty:
        print("데이터가 없습니다. --download 를 먼저 실행하세요.")
        sys.exit(1)
    run_all_models(df)


def run_all():
    run_download()
    run_analyze()
    run_train()


def main():
    parser = argparse.ArgumentParser(
        description="로또 번호 분석 프로젝트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--download", action="store_true", help="xlsx 초기 로드 + API 신규 회차 추가")
    parser.add_argument("--info",     action="store_true", help="DB 상태 출력")
    parser.add_argument("--analyze",  action="store_true", help="통계 분석 및 시각화")
    parser.add_argument("--train",    action="store_true", help="ML 모델 학습")
    parser.add_argument("--all",      action="store_true", help="전체 파이프라인 실행")

    args = parser.parse_args()

    if not any(vars(args).values()):
        parser.print_help()
        sys.exit(0)

    if args.download: run_download()
    if args.info:     run_info()
    if args.analyze:  run_analyze()
    if args.train:    run_train()
    if args.all:      run_all()


if __name__ == "__main__":
    main()
