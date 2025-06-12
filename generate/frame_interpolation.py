import numpy as np
import os

from utils import generate_gif

import torch
import random
import logging
import pathlib

from collections.abc import MutableMapping
def flatten_cfg(cfg):
    items = []
    for key, value in cfg.items():
        if isinstance(value, MutableMapping):
            items.extend(flatten_cfg(value).items())
        else:
            items.append((key, value))
    return dict(items)

import hydra
from omegaconf import DictConfig
import argparse
from random import randrange

def set_deterministic(seed):
    # Pytorch
    torch.manual_seed(seed)
    # Numpy
    np.random.seed(seed)
    # Random
    random.seed(seed)

import time
import pdb
@hydra.main(version_base=None, config_path="conf", config_name="base")
def main(cfg : DictConfig) -> None:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    cfg = flatten_cfg(cfg)
    args = argparse.Namespace(**cfg)

    ORF = args.ORF
    bs = args.bs
    exam_bs = args.exam_bs
    method = args.method
    print("[method, ORF, bs, exam_bs] = [{}, {}, {}, {}]\n\n".format(method, ORF, bs, exam_bs))
    mode = 'default'
    num = 10
    reverse_playing = False
    set_deterministic(0)
    target_dir = '/h/chchao/diff-yeast/generate/test_fig'
    target_path = pathlib.Path(target_dir)

    for rand_idx in range(num):
        filepath = target_path / ORF / method / str(rand_idx) / "selected_files"
        reference_mask_2 = np.load(filepath / 'reference_mask_2.npy', allow_pickle=True)
        reference_mask_1 = np.load(filepath / 'reference_mask_1.npy', allow_pickle=True)
        reference_mask_0 = np.load(filepath / 'reference_mask_0.npy', allow_pickle=True)
        reference_img_2 = np.load(filepath / 'reference_img_2.npy', allow_pickle=True)
        reference_img_1 = np.load(filepath / 'reference_img_1.npy', allow_pickle=True)
        reference_img_0 = np.load(filepath / 'reference_img_0.npy', allow_pickle=True)

        start = time.time()
        generate_gif(reference_mask_2, reference_mask_1, reference_mask_0, 
                    reference_img_2, reference_img_1, reference_img_0, 
                    filepath=os.path.join(target_path, ORF, method, str(rand_idx)),
                    filename=str(rand_idx) + '_video',
                    rotate_angle=0, flip_img=False, apply_mask=True, mode=mode,
                    reverse_playing=reverse_playing,
                    model_path='/h/chchao/film_net_fp16.pt')
        end = time.time()
        print("Successfully generate: {}".format(os.path.join(target_path, ORF, method, str(rand_idx))))
        print("Time (Frame Interpolation): {}".format(end - start))

if __name__ == '__main__':
    main()