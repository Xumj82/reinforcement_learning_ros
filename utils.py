import os
import logging
import logging.handlers
from datetime import datetime

def create_logger(name):
    log_dir = os.path.join(os.getcwd(),'log',datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger()

    logger.setLevel(logging.DEBUG)
    rf_handler =logging.handlers.RotatingFileHandler(
    log_dir+'/{}_all.log'.format(name), maxBytes=100, backupCount=5)
    rf_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    f_handler = logging.FileHandler(log_dir+'/{}_error.log'.format(name))
    f_handler.setLevel(logging.ERROR)
    f_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s[:%(lineno)d] - %(message)s"))

    logger.addHandler(rf_handler)
    logger.addHandler(f_handler)

    return logger