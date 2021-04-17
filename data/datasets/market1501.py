"""
This code is modified from Hao Luo's repository.
Paper: Bag of Tricks and A Strong Baseline for Deep Person Re-identification
https://github.com/michuanhaohao/reid-strong-baseline
"""

import glob
import re

import os.path as osp
from .bases import BaseImageDataset
import pickle
import torch
from data.datasets.dataset_loader import read_image
from PIL import Image
import numpy as np


class Market1501(BaseImageDataset):
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = 'market1501'

    def __init__(self, root='/home/haoluo/data', verbose=True, **kwargs):
        super(Market1501, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')
        self.mask_path = pickle.load(open(self.dataset_dir + '/image_mask_path_dict.pkl', 'rb'))
        self.mask_dir = osp.join(self.dataset_dir, 'Masks')

        self.att_train = pickle.load(open(osp.join(self.dataset_dir, 'attribute/market_train_att.pkl'), 'rb'))
        self.att_test = pickle.load(open(osp.join(self.dataset_dir, 'attribute/market_test_att.pkl'), 'rb'))

        self.att_name = pickle.load(open(osp.join(self.dataset_dir, 'attribute/market_att_label.pkl'), 'rb'))
        self.selected_attribute = [x for x in range(len(self.att_name))]

        self.inp = pickle.load(open(osp.join(self.dataset_dir, 'glove.pkl'), 'rb'))
        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> Market1501 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            img_id = img_path.split('/')[-1][:-4]
            mask_path = osp.join(self.mask_dir, self.mask_path[img_id])
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel:
                att = self.att_train[str(pid).zfill(4)]
                pid = pid2label[pid]
            else:
                if pid == 0:
                    att = [-1]*len(self.att_name)
                else:
                    att = self.att_test[str(pid).zfill(4)]
            dataset.append((img_path, pid, camid, att, self.inp, mask_path))


        return dataset

