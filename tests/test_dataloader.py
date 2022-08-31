import numpy as np

from data import Dataset

def test_dataloader():
    DATA_DIR = "data/Problem_01"
    dataset = Dataset(DATA_DIR)

    img = dataset.get_images(0)
    assert len(img.shape) == 3

    imgs = dataset.get_images([0, 2])
    assert len(imgs.shape) == 4
