"""Single-candidate serial scheduling with protocol-scoped cache."""

from collections.abc import Callable


class SearchScheduler:
    def __init__(self, run_one: Callable, cache: dict | None = None):
        self.run_one = run_one
        self.cache = cache if cache is not None else {}

    def evaluate(self, key: tuple, *args, **kwargs):
        if key in self.cache:
            return self.cache[key], True
        result = self.run_one(*args, **kwargs)
        self.cache[key] = result
        return result, False
