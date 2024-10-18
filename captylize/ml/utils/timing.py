import time
from functools import wraps


def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter_ns()
        result = func(*args, **kwargs)
        end_time = time.perf_counter_ns()
        execution_time_ms = end_time - start_time / 1_000_000
        return result, execution_time_ms

    return wrapper
