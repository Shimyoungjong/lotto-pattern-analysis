"""
로또 번호 분석 프로젝트 진입점.

사용법:
  python main.py --download   엑셀 다운로드 → DB 저장  (GitHub Actions / CI용)
  python main.py --load       로컬 CSV → DB 저장       (data/lotto_input.csv 사용)
  python main.py --info       DB 현재 상태 출력
  python main.py --analyze    통계 분석 + 시각화
  python main.py --train      ML 모델 학습
  python main.py --schedule   주간 로컬 스케줄러 실행
  python main.py --all        download → analyze → train 전체 파이프라인
"""

import argparse
import sys


def run_download(headless: bool = False):
    """동행복권 allWinExel 에서 엑셀 다운로드 후 DB 저장."""
    from src.downloader import download_and_parse
    from src.database import save_to_db

    print("=== 엑셀 다운로드 → DB 저장 ===")
    df = download_and_parse(headless=headless)
    save_to_db(df)


def run_load():
    """로컬 CSV 파일을 DB에 로드. (data/lotto_input.csv)"""
    from src.loader import load_csv
    from src.database import save_to_db

    print("=== CSV → DB 로드 ===")
    df = load_csv()
    save_to_db(df)


def run_info():
    from src.database import get_db_info

    info = get_db_info()
    if info["total"] == 0:
        print("DB가 비어 있습니다. --download 또는 --load 를 먼저 실행하세요.")
    else:
        print(f"DB 상태: 총 {info['total']}회차  |  {info['min_round']}~{info['max_round']}회차")


def run_analyze():
    from src.database import load_from_db
    from src.analysis import run_all_analysis

    print("=== 통계 분석 시작 ===")
    df = load_from_db()
    if df.empty:
        print("데이터가 없습니다. --download 또는 --load 를 먼저 실행하세요.")
        sys.exit(1)
    run_all_analysis(df)


def run_train():
    from src.database import load_from_db
    from src.models import run_all_models

    print("=== ML 모델 학습 시작 ===")
    df = load_from_db()
    if df.empty:
        print("데이터가 없습니다. --download 또는 --load 를 먼저 실행하세요.")
        sys.exit(1)
    run_all_models(df)


def run_schedule():
    from src.scheduler import start_scheduler
    start_scheduler()


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
    parser.add_argument("--download",  action="store_true", help="엑셀 다운로드 후 DB 저장")
    parser.add_argument("--headless",  action="store_true", help="브라우저를 헤드리스 모드로 실행 (CI용)")
    parser.add_argument("--load",      action="store_true", help="로컬 CSV를 DB에 로드")
    parser.add_argument("--info",      action="store_true", help="DB 상태 출력")
    parser.add_argument("--analyze",   action="store_true", help="통계 분석 및 시각화")
    parser.add_argument("--train",     action="store_true", help="ML 모델 학습")
    parser.add_argument("--schedule",  action="store_true", help="로컬 주간 스케줄러 실행")
    parser.add_argument("--all",       action="store_true", help="전체 파이프라인 실행")

    args = parser.parse_args()

    if not any(vars(args).values()):
        parser.print_help()
        sys.exit(0)

    if args.download:  run_download(headless=args.headless)
    if args.load:      run_load()
    if args.info:      run_info()
    if args.analyze:   run_analyze()
    if args.train:     run_train()
    if args.schedule:  run_schedule()
    if args.all:       run_all()


if __name__ == "__main__":
    main()
