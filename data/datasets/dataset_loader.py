"""
This code is modified from Hao Luo's repository.
Paper: Bag of Tricks and A Strong Baseline for Deep Person Re-identification
https://github.com/michuanhaohao/reid-strong-baseline
"""

import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch

def convert_to_bin_mask(mask):
    bin_mask = []
    for i in range(5):
        bin_mask.append((mask==i).astype(np.float))
    bin_mask.append((mask!=0).astype(np.float))
    bin_mask = np.stack(bin_mask)
    return bin_mask

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, att, inp, mask_path = self.dataset[index]
        img = read_image(img_path)
        mask = np.array(Image.open(mask_path).resize((256, 128), Image.BILINEAR))
        mask = torch.Tensor(convert_to_bin_mask(mask))
        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, img_path, att, inp, mask
