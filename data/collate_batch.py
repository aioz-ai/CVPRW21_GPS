"""
This code is modified from Hao Luo's repository.
Paper: Bag of Tricks and A Strong Baseline for Deep Person Re-identification
https://github.com/michuanhaohao/reid-strong-baseline
"""

import torch


def train_collate_fn(batch):
    imgs, pids, _, _, att, inp, masks = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)

    return torch.stack(imgs, dim=0), pids, att, inp, torch.stack(masks, dim=0)


def val_collate_fn(batch):
    imgs, pids, camids, path, att, inp, masks = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids, att, inp, torch.stack(masks, dim=0), path
