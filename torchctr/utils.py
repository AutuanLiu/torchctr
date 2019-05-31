import time
import pandas as pd
from pathlib import Path
import os


# 定义装饰器，监控运行时间
def timmer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        res = func(*args, **kwargs)
        stop_time = time.time()
        print(f'执行时刻: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}\n函数名: {func.__name__}\n执行时间: {(stop_time - start_time):.4f}s')
        return res

    return wrapper


def get_dat_data(fp):
    """读取 .dat 数据

    Args:
        fp (str or Path): 文件路径名
    """

    if not isinstance(fp, Path):
        fp = Path(fp)
    data = pd.read_csv(fp, sep='::', header=None, engine='python')
    return data


def num_cpus() -> int:
    "Get number of cpus"
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count()
