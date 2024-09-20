import numpy as np
import os

from dataloader import MyData
from utils import generate_gif, determine_center
from utils import iou_compute, similarity_compute, score_compute

from cellpose import utils, denoise, io
import tifffile as tiff
import pandas as pd
from ast import literal_eval

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

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

@hydra.main(version_base=None, config_path="conf", config_name="base")
def main(cfg : DictConfig) -> None:
    cfg = flatten_cfg(cfg)
    args = argparse.Namespace(**cfg)

    gene_name = args.gene_name
    img_idx = args.img_idx
    bs = args.bs
    exam_bs = args.exam_bs
    balance_fac = args.balance_fac

    loaded_df = pd.read_csv('/datasets/yeast-imgs/single_cell_annotations/group_by_protein_stage_rep1_filename_dict.csv')
    # for s in range(100):
    #     print(loaded_df['Gene Name'][s])
    # assert False
    df_list = []
    for i in range(6):
        print("Loading the ", str(i), "-th stage...")
        df = loaded_df[loaded_df['correctedMaxCycle_num'] == i]
        df = df[df['Gene Name'] == gene_name]["filename"].apply(literal_eval)
        df_list.append(df[df.index.values.astype(int)[0]])

    reference_mask = []
    reference_img_2 = []
    reference_img_1 = []
    reference_img_0 = []

    io.logger_setup()
    model = denoise.CellposeDenoiseModel(gpu=True, model_type="cyto3",restore_type="denoise_cyto3")

    for i in range(6): 
        print("Calculating the ", str(i), "-th stage...")
        all_choices = df_list[i]
        dataset = MyData(root_dir='/datasets/yeast-imgs/R1', data_list=all_choices, phase=i)
        data_loader = DataLoader(dataset, batch_size=bs, shuffle=False)

        if i == 0:
            imgs_channel_2 = None
            imgs_channel_1 = None
            imgs_channel_0 = None
            for imgs, labels in data_loader: 
                imgs_channel_2 = imgs[:,2,:,:]
                imgs_channel_1 = imgs[:,1,:,:]
                imgs_channel_0 = imgs[:,0,:,:]
                break
            masks_2, _, _, imgs_dn_2 = model.eval([imgs_channel_2[img_idx]], channels=[0,0])
            _, _, _, imgs_dn_1 = model.eval([imgs_channel_1[img_idx]], channels=[0,0])
            _, _, _, imgs_dn_0 = model.eval([imgs_channel_0[img_idx]], channels=[0,0])

            shift_list = sorted(determine_center(masks_2[0].squeeze()), key=lambda x: x[1])
            mask = shift_list[0][0]
            reference_mask.append(mask) # 64x64 np array
            reference_img_2.append((imgs_dn_2[0] * 255 / np.amax(imgs_dn_2[0])).astype(int)) # 1x64x64 np array
            reference_img_1.append((imgs_dn_1[0] * 255 / np.amax(imgs_dn_1[0])).astype(int)) # 1x64x64 np array
            reference_img_0.append((imgs_dn_0[0] * 255 / np.amax(imgs_dn_0[0])).astype(int)) # 1x64x64 np array
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
            _, _, _, imgs_dn_1 = model.eval(img_list_1, channels=[0,0])
            _, _, _, imgs_dn_0 = model.eval(img_list_0, channels=[0,0])
            
            rank_list = []
            for k in range(len(img_list_2)):
                print("Segmenting the ", str(k), "-th img...")
                shift_list = sorted(determine_center(masks_2[k].squeeze()), key=lambda x: x[1])
                mask = shift_list[0][0] if (i < 3 or len(shift_list)==1) else (shift_list[0][0] + shift_list[1][0])
                img_2 = (imgs_dn_2[k] * 255 / np.amax(imgs_dn_2[k])).astype(int)
                img_1 = (imgs_dn_1[k] * 255 / np.amax(imgs_dn_1[k])).astype(int)
                img_0 = (imgs_dn_0[k] * 255 / np.amax(imgs_dn_0[k])).astype(int)
                size = np.count_nonzero(mask)
                weight = [0.4,0.4,0.2]
                balance_fac = 0.25#1.0 if i in [1,5] else 0.25
                score = score_compute(mask_list=[mask, reference_mask[-1]],
                                        img_2_list=[img_2, reference_img_2[-1]],
                                        img_1_list=[img_1, reference_img_1[-1]],
                                        img_0_list=[img_0, reference_img_0[-1]],
                                        balance_fac=balance_fac, weight=weight)
                rank_list.append([size, score, mask, img_2, img_1, img_0])

            rank_list = sorted(rank_list, key=lambda x: x[0])
            splits = 1 #(len(rank_list) // 100) + 1
            num_each_split = 1 #(len(rank_list) // 90) + 1 if i in [1,5] else 1
            if i == 5 and i == 4:
                num_each_split += 1
            for k in range(splits):
                rank_list_sorted = sorted(rank_list[k*(len(rank_list)//splits) : (k+1)*(len(rank_list)//splits)], key=lambda x: x[1], reverse=True)
                for j in range(num_each_split):
                    reference_mask.append(rank_list_sorted[j][2])
                    reference_img_2.append(rank_list_sorted[j][3]) # 1x64x64 np array
                    reference_img_1.append(rank_list_sorted[j][4]) # 1x64x64 np array
                    reference_img_0.append(rank_list_sorted[j][5]) # 1x64x64 np array

    generate_gif(reference_mask, reference_img_2, 
                reference_img_1, reference_img_0, 
                filename="array_"+str(gene_name)+"_"+str(img_idx))
    print("Successfully generate: ", "array_"+str(gene_name)+"_"+str(img_idx))

if __name__ == '__main__':
    main()