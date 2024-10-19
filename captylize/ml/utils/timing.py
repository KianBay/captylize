import time
from functools import wraps
from typing import Callable, TypeVar, Tuple, Any

T = TypeVar("T")


def measure_time(func: Callable[..., T]) -> Callable[..., Tuple[T, float]]:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Tuple[T, float]:
        start_time = time.perf_counter_ns()
        result = func(*args, **kwargs)
        end_time = time.perf_counter_ns()
        execution_time_ms = (end_time - start_time) / 1_000_000
        return result, execution_time_ms

    return wrapper
