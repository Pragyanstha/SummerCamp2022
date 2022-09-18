import sys
print(sys.path)

import numpy as np
import configargparse

from args import get_config
from data import Dataset
from algorithms import get_algo

def main(config):
    print(config)

    dataset = Dataset(config.data_dir, config.result_dir,  fps=config.fps)
    algo = get_algo(config)
    ## Main Algorithm here
    results = algo(dataset) # returs some dummpy values for now
    print(results)
    dataset.write_results(results)
    ##

    # evaluate the results
    eval_res = dataset.evaluate(results)

if __name__ == "__main__":
    config = get_config()
    main(config)
    print("dekitayo")
