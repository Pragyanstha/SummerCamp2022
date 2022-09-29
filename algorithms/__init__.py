from lib2to3.pytree import Base
from algorithms.baseline import Baseline
from algorithms.baseline_ML import Baseline_ML

ALGOS = {
    "baseline": Baseline,
    "baseline_ML": Baseline_ML
}

def get_algo(config):
    return ALGOS[config.algo](config)

