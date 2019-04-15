import time


# 定义装饰器，监控运行时间
def timmer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        res = func(*args, **kwargs)
        stop_time = time.time()
        print(f'执行时刻: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}\n函数名: {func.__name__}\n执行时间: {(stop_time - start_time):.4f}s')
        return res

    return wrapper
