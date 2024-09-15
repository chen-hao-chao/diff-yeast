import numpy as np
import time, os, sys
from urllib.parse import urlparse
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200
from cellpose import utils, denoise, io
import tifffile as tiff
import pandas as pd
from ast import literal_eval 
from dataloader import MyData
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision
import torch.nn.functional as F
from PIL import Image

def merge(img_1, img_2, img_3, mask):
    # img_1: (1,64,64) is the base 
    # img_2: (1,64,64) takes the R channel
    # img_3: (1,64,64) takes the G channel
    # mask: (1,64,64) is a segmentation mask
    # ---------
    img_1 = np.expand_dims(img_1.squeeze()*mask, axis=0)
    img_2 = np.expand_dims(img_2.squeeze()*mask, axis=0)
    img_3 = np.expand_dims(img_3.squeeze()*mask, axis=0)

    img_1_ = np.vstack([img_1, img_1, img_1])
    img_2_ = np.vstack([img_2, np.zeros(img_2.shape), np.zeros(img_2.shape)])
    img_3_ = np.vstack([np.zeros(img_3.shape), img_3, np.zeros(img_3.shape)])
    return np.dstack(img_1_*0.2 + img_2_*0.4 + img_3_*0.4).astype(int)

def iou_compute(outputs: np.array, labels: np.array):
    # source: https://www.kaggle.com/code/iezepov/fast-iou-scoring-metric-in-pytorch-and-numpy
    SMOOTH = 1e-6
    intersection = outputs * labels
    intersection = intersection[intersection != 0].sum()
    union = outputs + labels
    union = union[union != 0].sum()
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    return iou

from skimage.metrics import structural_similarity
def similarity_compute(img_1: np.array, img_2: np.array):
    # img_1: np array (64x64x1)
    # img_2: np array (64x64x1)
    score = structural_similarity(img_1.squeeze()/255, img_2.squeeze()/255, data_range=1.0)
    return score

def determine_center(seg):
    # seg: (numpy) a map with indeces
    max_idx = np.amax(seg)
    shift_list = []
    mask = np.zeros((64,64))
    min_shift = 10000
    for i in range(1,int(max_idx)+1):
        z = np.zeros((64,64))
        z[seg == i] = 1
        
        shift = 0
        # https://stackoverflow.com/questions/51716954/how-to-center-the-nonzero-values-within-2d-numpy-array
        for k in range(2):
            nonempty = np.nonzero(np.any(z, axis=1-k))[0]
            first, last = nonempty.min(), nonempty.max()
            shift += np.abs(z.shape[k] - first - last)//2
        shift_list.append([z, shift])
    if len(shift_list) == 0:
        shift_list.append([np.zeros((64,64)), 0])
    return shift_list

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
        # print(df[df.index.values.astype(int)[0]])
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
            

            max_score = 0
            max_idx = 0
            max_mask = None

            for j in range(len(img_list_2)):
                print("Segmenting the ", str(j), "-th img...")
                shift_list = sorted(determine_center(masks_2[j].squeeze()), key=lambda x: x[1])
                mask = shift_list[0][0] if (i < 2 or len(shift_list)==1) else (shift_list[0][0] + shift_list[1][0])
                iou = iou_compute(reference_mask[i-1], mask)
                ssim_2 = similarity_compute( reference_img_2[i-1], (imgs_dn_2[j] * 255 / np.amax(imgs_dn_2[j])).astype(int) )
                ssim_1 = similarity_compute( reference_img_1[i-1], (imgs_dn_1[j] * 255 / np.amax(imgs_dn_1[j])).astype(int) )
                ssim_0 = similarity_compute( reference_img_0[i-1], (imgs_dn_0[j] * 255 / np.amax(imgs_dn_0[j])).astype(int) )
                score = iou + balance_fac*((ssim_2 + ssim_1 + ssim_0) / 3)
                print(iou, ssim_2, ssim_1, ssim_0)
                if max_score < score:
                    max_score = score
                    max_mask = mask
                    max_idx = j
                    
            reference_mask.append(max_mask)
            reference_img_2.append((imgs_dn_2[max_idx] * 255 / np.amax(imgs_dn_2[max_idx])).astype(int)) # 1x64x64 np array
            reference_img_1.append((imgs_dn_1[max_idx] * 255 / np.amax(imgs_dn_1[max_idx])).astype(int)) # 1x64x64 np array
            reference_img_0.append((imgs_dn_0[max_idx] * 255 / np.amax(imgs_dn_0[max_idx])).astype(int)) # 1x64x64 np array

    gif = []
    for i in range(6):
        merged_img = merge(reference_img_2[i], reference_img_1[i], reference_img_0[i], reference_mask[i])
        print(merged_img.shape)
        gif.append(Image.fromarray(merged_img.astype(np.uint8), mode='RGB'))
    gif[0].save("array_"+str(gene_name)+"_"+str(img_idx)+".gif", save_all=True, append_images=gif[1:], duration=100, loop=0)

if __name__ == '__main__':
    main()