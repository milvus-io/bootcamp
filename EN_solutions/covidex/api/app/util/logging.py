import logging
import time
from logging.handlers import TimedRotatingFileHandler

import pkg_resources

from app.settings import settings


def build_timed_logger(name: str, filename: str) -> logging.Logger:
    '''
    Returns a logger that creates appends a new log file daily
    '''
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not settings.testing:
        path = f"{settings.log_path}/{filename}"
        handler = TimedRotatingFileHandler(path, when="d", interval=1, utc=True)
        logger.addHandler(handler)
    return logger
