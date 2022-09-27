
import os
import sys
import logging
import logging.handlers
import functools
from pathlib import Path
logger_initialized = {}

@functools.lru_cache()
def get_logger(name:str ='root', log_level=logging.INFO):

    # logger 생성
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger

    # log 출력 형태 정의
    formatter = logging.Formatter(
        '[%(asctime)s] %(name)s %(levelname)s: %(message)s',
        datefmt="%Y/%m/%d %H:%M:%S")

    # logfile 저장위치
    current_dir = os.path.dirname(os.path.realpath(__file__))
    log_dir = os.path.join(current_dir, "logs")
    Path(log_dir).mkdir(exist_ok=True, parents=True)

    # streaming log
    # stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter) # formatter 를 넣어돠야되나... 흠
    # stream_handler.setLevel(log_level)
    logger.addHandler(stream_handler)

    # 월요일 기준으로 로그파일 저장
    file_handler = logging.handlers.TimedRotatingFileHandler(filename=os.path.join(log_dir, "speech2text.log"), when="W0", interval=1, encoding="utf-8")
    file_handler.suffix = "%Y%m%d"
    file_handler.setFormatter(formatter)
    # file_handler.setLevel(log_level)
    logger.addHandler(file_handler)
    logger.setLevel(level=log_level)

    return logger
