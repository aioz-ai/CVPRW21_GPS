"""
This code is modified from Hao Luo's repository.
Paper: Bag of Tricks and A Strong Baseline for Deep Person Re-identification
https://github.com/michuanhaohao/reid-strong-baseline
-------------------------------------------------------------------------------
Edited by Binh X. Nguyen
Paper: Graph-based Person Signature for Person Re-Identifications
https://github.com/aioz-ai/CVPRW21_GPS
"""

from .person_graph import PersonGraph

def build_model(cfg, num_classes):
    model = PersonGraph(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT, cfg.MODEL.BACKBONE, cfg.MODEL.PRETRAIN_CHOICE, cfg.DATASETS.NAMES, cfg.MODEL.PART, cfg.MODEL.ATT)
    return model
