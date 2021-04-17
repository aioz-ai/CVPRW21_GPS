"""
This code is modified from Hao Luo's repository.
Paper: Bag of Tricks and A Strong Baseline for Deep Person Re-identification
https://github.com/michuanhaohao/reid-strong-baseline
"""
from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501
from .dataset_loader import ImageDataset

__factory = {
    'market1501': Market1501,
    'dukemtmc': DukeMTMCreID,
}


def get_names():
    return __factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)
