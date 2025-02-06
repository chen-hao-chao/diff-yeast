import numpy as np
import os

from dataloader import CellDataLoader
from utils import generate_gif, determine_center, visualize, add_outline, rotate_to_normalize, flip_to_normalize, aggregate_masks
from utils import iou_compute, similarity_compute, score_compute, angle_sim_compute
from utils import normalize_intensity, merge, pseudo_segmentation, shift_intensity

import tifffile as tiff
import pandas as pd
from ast import literal_eval

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import random

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


@hydra.main(version_base=None, config_path="conf", config_name="base")
def main(cfg : DictConfig) -> None:
    cfg = flatten_cfg(cfg)
    args = argparse.Namespace(**cfg)
    ORF = args.ORF
    bs = args.bs
    exam_bs = args.exam_bs
    mode = 'default' #'structure' #'default'
    avg = 0.125 #0.2 #0.125
    num = 5
    reverse_playing = True
    set_deterministic(0)

    target_dir = './test_fig' #'/datasets/yeast-imgs/gt_videos/R1'
    target_path = os.path.join(target_dir, ORF)
    if not os.path.exists(target_path):
        os.makedirs(target_path)
        print("Create a new directory: ", target_path)
    else:
        print("Directory exists.")
    
    root_dir_no_gfp='/fs01/datasets/yeast-imgs/cellcycle_single_cell_crops/128/select_proteins/no_GFP/R1'
    df_no_gfp_sublist = [f for f in os.listdir(root_dir_no_gfp) if f.split('.')[1]=='tiff']
    dataset_no_gfp = CellDataLoader(root_dir=root_dir_no_gfp, data_list=df_no_gfp_sublist)
    data_loader_no_gfp = DataLoader(dataset_no_gfp, batch_size=len(df_no_gfp_sublist), shuffle=False)

    mean_value = 0
    for imgs in data_loader_no_gfp:
        imgs_channel_no_gfp_0 = imgs[:,0,:,:].numpy()
        imgs_channel_no_gfp_0 = imgs_channel_no_gfp_0
        mean_value = np.mean(imgs_channel_no_gfp_0[imgs_channel_no_gfp_0 != 0])
        print(mean_value)
        break

    loaded_df = pd.read_csv('/datasets/yeast-imgs/single_cell_annotations/group_by_protein_stage_rep1_filename_dict.csv')
    root_dir='/fs01/datasets/yeast-imgs/cellcycle_single_cell_crops/128/select_proteins/R1'
    
    df_list = []
    for i in range(6):
        print("Loading the ", str(i), "-th stage...")
        df = loaded_df[loaded_df['correctedMaxCycle_num'] == i]
        df = df[df['ORF'] == ORF]["filename"].apply(literal_eval)
        file = df[df.index.values.astype(int)[0]]
        df_list.append(file)

    for rand_idx in range(num):
        # rand number
        rnd_0 = randrange(len(df_list[5])) if reverse_playing else randrange(len(df_list[0]))
        rnd_1 = random.uniform(-1, 1)
        rnd_2 = 0 #random.uniform(-1, 1)
        rnd_3 = random.uniform(-1, 1)
        rnd_4 = randrange(3) if reverse_playing else randrange(7) # -> 3

        img_idx = rnd_0
        balance_fac = args.balance_fac + rnd_1 * 0.2
        balance_fac_cat = 0.1
        angle = rnd_2 * 180
        flip = bool(rnd_3>0)
        rnk = rnd_4
        
        print("img_idx: ", img_idx)
        print("balance_fac: ", balance_fac)
        print("balance_fac_cat: ", balance_fac_cat)
        print("angle: ", angle)
        print("flip: ", flip)
        print("rnk: ", rnk)

        reference_mask_2 = []
        reference_mask_1 = []
        reference_mask_0 = []
        reference_img_2 = []
        reference_img_1 = []
        reference_img_0 = []

        traverse_list = [5,4,3,2,1,0] if reverse_playing else [0,1,2,3,4,5]
        for i in traverse_list: 
            print("Calculating the ", str(i), "-th stage...")
            all_choices = df_list[i]
            dataset = CellDataLoader(root_dir=root_dir, data_list=all_choices)
            data_loader = DataLoader(dataset, batch_size=bs, shuffle=False)
            if (i == 5 and reverse_playing) or (i == 0 and not reverse_playing):
                imgs_channel_2 = None
                imgs_channel_1 = None
                imgs_channel_0 = None
                batch_count = 0
                for imgs in data_loader:
                    imgs_channel_2 = imgs[:,2,:,:].numpy()
                    imgs_channel_1 = imgs[:,1,:,:].numpy()
                    imgs_channel_0 = imgs[:,0,:,:].numpy()
                    if batch_count == (img_idx // bs):
                        break
                    else:
                        batch_count = batch_count + 1

                masks_2, imgs_dn_2 = pseudo_segmentation([imgs_channel_2[img_idx % bs]])
                masks_1, imgs_dn_1 = pseudo_segmentation([imgs_channel_1[img_idx % bs]])
                masks_0, imgs_dn_0 = pseudo_segmentation([imgs_channel_0[img_idx % bs]])

                mask_2 = masks_2[0]
                mask_1 = masks_1[0]
                mask_0 = masks_0[0]

                img_2 = normalize_intensity(imgs_dn_2[0], avg=avg)
                img_1 = normalize_intensity(imgs_dn_1[0], avg=avg)
                img_0 = normalize_intensity(shift_intensity(imgs_dn_0[0], mean_value), avg=avg)

                img_2, img_1, img_0, mask_2, mask_1, mask_0 = rotate_to_normalize(img_2, img_1, img_0, mask_2, mask_1, mask_0)

                reference_mask_2.append(mask_2) # 64x64 np array
                reference_mask_1.append(mask_1) # 64x64 np array
                reference_mask_0.append(mask_0) # 64x64 np array
                reference_img_2.append(img_2) # 1x64x64 np array
                reference_img_1.append(img_1) # 1x64x64 np array
                reference_img_0.append(img_0) # 1x64x64 np array
                continue
            else:
                img_list_2 = []
                img_list_1 = []
                img_list_0 = []
                idx = 0
                for imgs in data_loader:
                    imgs_channel_2 = imgs[:,2,:,:].numpy()
                    imgs_channel_1 = imgs[:,1,:,:].numpy()
                    imgs_channel_0 = imgs[:,0,:,:].numpy()
                    for k in range(imgs.shape[0]):
                        img_list_2.append(imgs_channel_2[k,:,:])
                        img_list_1.append(imgs_channel_1[k,:,:])
                        img_list_0.append(imgs_channel_0[k,:,:])
                    idx += 1
                    if idx == exam_bs:
                        break

                masks_2, imgs_dn_2 = pseudo_segmentation(img_list_2)
                masks_1, imgs_dn_1 = pseudo_segmentation(img_list_1)
                masks_0, imgs_dn_0 = pseudo_segmentation(img_list_0)

                rank_list = []
                for k in range(len(img_list_2)):
                    print("Segmenting the ", str(k), "-th img...")
                    mask_2 = masks_2[k]
                    mask_1 = masks_1[k]
                    mask_0 = masks_0[k]

                    img_2 = normalize_intensity(imgs_dn_2[k], avg=avg)
                    img_1 = normalize_intensity(imgs_dn_1[k], avg=avg)
                    img_0 = normalize_intensity(shift_intensity(imgs_dn_0[k], mean_value), avg=avg)
                
                    img_2, img_1, img_0, mask_2, mask_1, mask_0 = rotate_to_normalize(img_2, img_1, img_0, mask_2, mask_1, mask_0)

                    size = np.count_nonzero(aggregate_masks(mask_2, mask_1, mask_0))
                    weight = [0.45,0.45,0.1]
                    weight_iou = [0.0,1.0,0.0]
                    score = score_compute(mask_list_2=[mask_2, reference_mask_2[-1]],
                                        mask_list_1=[mask_1, reference_mask_1[-1]],
                                        mask_list_0=[mask_0, reference_mask_0[-1]],
                                        img_2_list=[img_2, reference_img_2[-1]],
                                        img_1_list=[img_1, reference_img_1[-1]],
                                        img_0_list=[img_0, reference_img_0[-1]],
                                        balance_fac=balance_fac,
                                        balance_fac_cat=balance_fac_cat,
                                        weight=weight,
                                        weight_iou=weight_iou)
                    rank_list.append([size, score, mask_2, mask_1, mask_0, img_2, img_1, img_0])

                    img_2, img_1, img_0, mask_2, mask_1, mask_0 = flip_to_normalize(img_2, img_1, img_0, mask_2, mask_1, mask_0)
                    score = score_compute(mask_list_2=[mask_2, reference_mask_2[-1]],
                                        mask_list_1=[mask_1, reference_mask_1[-1]],
                                        mask_list_0=[mask_0, reference_mask_0[-1]],
                                        img_2_list=[img_2, reference_img_2[-1]],
                                        img_1_list=[img_1, reference_img_1[-1]],
                                        img_0_list=[img_0, reference_img_0[-1]],
                                        balance_fac=balance_fac,
                                        balance_fac_cat=balance_fac_cat,
                                        weight=weight,
                                        weight_iou=weight_iou)
                    rank_list.append([size, score, mask_2, mask_1, mask_0, img_2, img_1, img_0])

                rank_list = sorted(rank_list, key=lambda x: x[0])
                splits = 1
                num_frames = 1
                rank_list_sorted_size = sorted(rank_list, key=lambda x: x[0])
                for k in range(splits):
                    rank_list_sorted = sorted(rank_list_sorted_size[k*(len(rank_list)//splits) : (k+1)*(len(rank_list)//splits)], key=lambda x: x[1], reverse=True)
                    for n in range(num_frames):
                        reference_mask_2.append(rank_list_sorted[rnk+n][2])
                        reference_mask_1.append(rank_list_sorted[rnk+n][3])
                        reference_mask_0.append(rank_list_sorted[rnk+n][4])
                        reference_img_2.append(rank_list_sorted[rnk+n][5]) # 1x64x64 np array
                        reference_img_1.append(rank_list_sorted[rnk+n][6]) # 1x64x64 np array
                        reference_img_0.append(rank_list_sorted[rnk+n][7]) # 1x64x64 np array

        generate_gif(reference_mask_2, reference_mask_1, reference_mask_0, 
                    reference_img_2, reference_img_1, reference_img_0, 
                    filename=os.path.join(target_path, str(rand_idx)+'_'+mode),
                    rotate_angle=angle, flip_img=flip, apply_mask=True, mode=mode,
                    reverse_playing=reverse_playing)
        print("Successfully generate: ", os.path.join(target_path, str(rand_idx)))

if __name__ == '__main__':
    main()