import numpy as np
import os

from dataloader import MyData
from utils import generate_gif, determine_center, visualize, add_outline, rotate_to_normalize, flip_to_normalize, aggregate_masks
from utils import iou_compute, similarity_compute, score_compute, angle_sim_compute

from cellpose import utils, denoise, io
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

@hydra.main(version_base=None, config_path="conf", config_name="base")
def main(cfg : DictConfig) -> None:
    cfg = flatten_cfg(cfg)
    args = argparse.Namespace(**cfg)
    ORF = args.ORF
    bs = args.bs
    exam_bs = args.exam_bs
    split_stage = args.split_stage

    target_dir = './test_fig' #'/datasets/yeast-imgs/gt_videos/R1'
    target_path = os.path.join(target_dir, ORF)
    if not os.path.exists(target_path):
        os.makedirs(target_path)
        print("Create a new directory: ", target_path)
    else:
        print("Directory exists.")

    loaded_df = pd.read_csv('/datasets/yeast-imgs/single_cell_annotations/group_by_protein_stage_rep1_filename_dict.csv')
    # for s in range(100):
    #     print(loaded_df['Gene Name'][s])
    # print(loaded_df)
    # assert False
    df_list = []
    for i in range(6):
        print("Loading the ", str(i), "-th stage...")
        df = loaded_df[loaded_df['correctedMaxCycle_num'] == i]
        df = df[df['ORF'] == ORF]["filename"].apply(literal_eval)
        df_list.append(df[df.index.values.astype(int)[0]])

    io.logger_setup()
    model = denoise.CellposeDenoiseModel(gpu=True, model_type="cyto3",restore_type="denoise_cyto3")

    for rand_idx in range(1): #len(df_list[0])
        # rand number
        rnd_0 = 0 #randrange(len(df_list[0]))
        rnd_1 = 0 #random.uniform(-1, 1)
        rnd_2 = 0 #random.uniform(-1, 1)
        rnd_3 = 0 #random.uniform(-1, 1)
        rnd_4 = 0 #randrange(2)

        img_idx = rnd_0
        balance_fac = args.balance_fac + rnd_1 * 0.2
        balance_fac_cat = 0.25
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

        for i in range(6): 
            print("Calculating the ", str(i), "-th stage...")
            all_choices = df_list[i]
            dataset = MyData(root_dir='/datasets/yeast-imgs/R1', data_list=all_choices, phase=0)
            data_loader = DataLoader(dataset, batch_size=bs, shuffle=False)
            if i == 0:
                imgs_channel_2 = None
                imgs_channel_1 = None
                imgs_channel_0 = None
                batch_count = 0
                for imgs, labels in data_loader:
                    imgs_channel_2 = imgs[:,2,:,:]
                    imgs_channel_1 = imgs[:,1,:,:]
                    imgs_channel_0 = imgs[:,0,:,:]
                    if batch_count == (img_idx // bs):
                        break
                    else:
                        batch_count = batch_count + 1
                masks_2, _, _, imgs_dn_2 = model.eval([imgs_channel_2[img_idx % bs]], channels=[0,0])
                masks_1, _, _, imgs_dn_1 = model.eval([imgs_channel_1[img_idx % bs]], channels=[0,0])
                masks_0, _, _, imgs_dn_0 = model.eval([imgs_channel_0[img_idx % bs]], channels=[0,0])

                shift_list = sorted(determine_center(masks_2[0].squeeze()), key=lambda x: x[1])
                mask_2 = shift_list[0][0]
                shift_list = sorted(determine_center(masks_1[0].squeeze()), key=lambda x: x[1])
                mask_1 = shift_list[0][0]
                shift_list = sorted(determine_center(masks_0[0].squeeze()), key=lambda x: x[1])
                mask_0 = shift_list[0][0]

                # lined = add_outline((imgs_dn_2[0] * 255 / np.amax(imgs_dn_2[0])).astype(int), mask_2)
                # visualize(lined.squeeze()*mask_2, "test.png")
                # assert False

                img_2 = (imgs_dn_2[0] * 255 / np.amax(imgs_dn_2[0])).astype(np.uint8)
                img_1 = (imgs_dn_1[0] * 255 / np.amax(imgs_dn_1[0])).astype(np.uint8)
                img_0 = (imgs_dn_0[0] * 255 / np.amax(imgs_dn_0[0])).astype(np.uint8)
                
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
                for imgs, labels in data_loader:
                    imgs_channel_2 = imgs[:,2,:,:]
                    imgs_channel_1 = imgs[:,1,:,:]
                    imgs_channel_0 = imgs[:,0,:,:]
                    for k in range(imgs.shape[0]):
                        img_list_2.append(imgs_channel_2[k,:,:])
                        img_list_1.append(imgs_channel_1[k,:,:])
                        img_list_0.append(imgs_channel_0[k,:,:])
                    idx += 1
                    if idx == exam_bs:
                        break
                
                masks_2, _, _, imgs_dn_2 = model.eval(img_list_2, channels=[0,0])
                masks_1, _, _, imgs_dn_1 = model.eval(img_list_1, channels=[0,0])
                masks_0, _, _, imgs_dn_0 = model.eval(img_list_0, channels=[0,0])

                rank_list = []
                for k in range(len(img_list_2)):
                    print("Segmenting the ", str(k), "-th img...")
                    shift_list = sorted(determine_center(masks_2[k].squeeze()), key=lambda x: x[1])
                    mask_2 = shift_list[0][0] if (i < split_stage or len(shift_list)==1) else np.clip(shift_list[0][0] + shift_list[1][0], 0, 1)
                    shift_list = sorted(determine_center(masks_1[k].squeeze()), key=lambda x: x[1])
                    mask_1 = shift_list[0][0] if (i < split_stage or len(shift_list)==1) else np.clip(shift_list[0][0] + shift_list[1][0], 0, 1)
                    shift_list = sorted(determine_center(masks_0[k].squeeze()), key=lambda x: x[1])
                    mask_0 = shift_list[0][0] if (i < split_stage or len(shift_list)==1) else np.clip(shift_list[0][0] + shift_list[1][0], 0, 1)
                    
                    img_2 = (imgs_dn_2[k] * 255 / np.amax(imgs_dn_2[k])).astype(np.uint8)
                    img_1 = (imgs_dn_1[k] * 255 / np.amax(imgs_dn_1[k])).astype(np.uint8)
                    img_0 = (imgs_dn_0[k] * 255 / np.amax(imgs_dn_0[k])).astype(np.uint8)
                
                    img_2, img_1, img_0, mask_2, mask_1, mask_0 = rotate_to_normalize(img_2, img_1, img_0, mask_2, mask_1, mask_0)

                    size = np.count_nonzero(aggregate_masks(mask_2, mask_1, mask_0))
                    weight = [0.4,0.4,0.2]
                    weight_iou = [0.0,0.0,1.0]
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
                    weight = [0.4,0.4,0.2]
                    weight_iou = [0.0,0.0,1.0]
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
                splits = 2
                num_frames = 2
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
                    filename=os.path.join(target_path, str(rand_idx)),
                    rotate_angle=angle, flip_img=flip)
        print("Successfully generate: ", os.path.join(target_path, str(rand_idx)))

if __name__ == '__main__':
    main()