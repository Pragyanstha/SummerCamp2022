from lib2to3.pytree import Base
from algorithms.baseline import Baseline
from algorithms.mcftracker import MCFTracker

ALGOS = {
    "baseline": Baseline,
    "mcftracker": MCFTracker
}

def get_algo(config):
    return ALGOS[config.algo](config)

