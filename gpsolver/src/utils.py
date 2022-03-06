import logging

logger = logging.getLogger()


def set_logger(log_file, debug):
    logger.setLevel(logging.DEBUG)
    msg_fmt = "%(asctime)s - %(levelname)-5s - %(name)s -   %(message)s"
    date_fmt = "%m/%d/%Y %H:%M:%S"

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG if debug else logging.INFO)

    formatter = logging.Formatter(fmt=msg_fmt, datefmt=date_fmt)
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
