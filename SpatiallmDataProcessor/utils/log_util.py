import logging

def init_logger(filename):
    """初始化日志"""
    logging.basicConfig(
        filename=filename,
        level=logging.DEBUG,
        format='%(asctime)s | %(levelname)s | %(message)s',
        filemode='w'
    )
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
