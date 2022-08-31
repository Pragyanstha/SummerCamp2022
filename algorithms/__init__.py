from lib2to3.pytree import Base
from algorithms.baseline import Baseline

ALGOS = {
    "baseline": Baseline
}

def get_algo(config):
    return ALGOS[config.algo](config)

