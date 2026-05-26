import logging
from pathlib import Path


def configure_logging() -> None:
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        handlers=[
            logging.FileHandler("logs/fii_analytics.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

