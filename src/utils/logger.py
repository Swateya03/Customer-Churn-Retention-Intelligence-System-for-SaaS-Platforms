import logging
import os

def get_logger(name):
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        filename="logs/data_pipeline.log",
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger(name)
