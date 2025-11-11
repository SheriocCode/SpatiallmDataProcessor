import os

def make_dirs(*dirs):
    """批量创建目录"""
    for dir in dirs:
        os.makedirs(dir, exist_ok=True)
