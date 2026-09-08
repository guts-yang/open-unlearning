from evals.base import Evaluator


class LeakRUEvaluator(Evaluator):
    def __init__(self, eval_cfg, **kwargs):
        super().__init__("LeakRU", eval_cfg, **kwargs)
