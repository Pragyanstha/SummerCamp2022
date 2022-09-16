import numpy as np

from data import Dataset

def test_dataloader():
    DATA_DIR = "/workspace/SummerCamp2022/data/Problem_01"
    RESULT_DIR = "/workspace/SummerCamp2022/results"

    dataset = Dataset(DATA_DIR, RESULT_DIR, fps=5)

    img = dataset.get_images(0)
    assert len(img.shape) == 3

    imgs = dataset.get_images([0, 2])
    assert len(imgs.shape) == 4
