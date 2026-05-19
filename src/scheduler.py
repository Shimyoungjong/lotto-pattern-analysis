"""
APScheduler를 사용해 매주 토요일 밤 DB를 자동 갱신하는 모듈.

업데이트 흐름:
  1. 사용자가 최신 CSV를 다운받아 data/lotto_input.csv 로 교체
  2. 스케줄러가 매주 토요일 21:10 에 자동으로 CSV → DB 갱신 실행
"""

import logging
from pathlib import Path
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from src.loader import load_csv, INPUT_CSV
from src.database import save_to_db, get_db_info

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def update_job():
    """매주 실행되는 CSV → DB 갱신 작업."""
    logger.info("===== 로또 데이터 자동 업데이트 시작 =====")

    if not INPUT_CSV.exists():
        logger.warning(f"CSV 파일 없음: {INPUT_CSV}")
        logger.warning("data/lotto_input.csv 를 최신 파일로 교체한 뒤 스케줄러를 재시작하세요.")
        return

    try:
        df = load_csv(INPUT_CSV)
        save_to_db(df)
        info = get_db_info()
        logger.info(f"갱신 완료 — 총 {info['total']}회차 ({info['min_round']}~{info['max_round']}회차)")
    except Exception as e:
        logger.error(f"업데이트 실패: {e}")

    logger.info("===== 자동 업데이트 종료 =====")


def start_scheduler():
    """
    스케줄러를 시작한다.
    매주 토요일 오후 9시 10분에 실행 (추첨 후 여유 시간 확보).
    """
    scheduler = BlockingScheduler(timezone="Asia/Seoul")

    scheduler.add_job(
        update_job,
        trigger=CronTrigger(day_of_week="sat", hour=21, minute=10, timezone="Asia/Seoul"),
        id="lotto_weekly_update",
        name="로또 주간 업데이트",
        replace_existing=True,
    )

    logger.info("스케줄러 시작 (매주 토요일 21:10 자동 실행)")
    logger.info("업데이트 전 data/lotto_input.csv 를 최신 파일로 교체해 주세요.")
    logger.info("종료하려면 Ctrl+C 를 누르세요.")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("스케줄러 종료.")
