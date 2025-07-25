import logging


def get_logger(logging_level=logging.INFO):
    logging.basicConfig(level=logging_level,
                        format="'%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    return logging.getLogger()
