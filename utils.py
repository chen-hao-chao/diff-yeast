import numpy as np
from skimage.metrics import structural_similarity

def merge(img_2, img_1, img_0, mask):
    # img_2: (1,64,64) is the base 
    # img_1: (1,64,64) takes the R channel
    # img_0: (1,64,64) takes the G channel
    # mask: (1,64,64) is a segmentation mask
    # ---------
    img_2 = np.expand_dims(img_2.squeeze()*mask, axis=0)
    img_1 = np.expand_dims(img_1.squeeze()*mask, axis=0)
    img_0 = np.expand_dims(img_0.squeeze()*mask, axis=0)

    img_2_ = np.vstack([img_2, img_2, img_2])
    img_1_ = np.vstack([img_1, np.zeros(img_1.shape), np.zeros(img_1.shape)])
    img_0_ = np.vstack([np.zeros(img_0.shape), img_0, np.zeros(img_0.shape)])
    return np.dstack(img_2_*0.1 + img_1_*0.45 + img_0_*0.45).astype(int)

def iou_compute(outputs: np.array, labels: np.array):
    # source: https://www.kaggle.com/code/iezepov/fast-iou-scoring-metric-in-pytorch-and-numpy
    SMOOTH = 1e-6
    intersection = outputs * labels
    intersection = intersection[intersection != 0].sum()
    union = outputs + labels
    union = union[union != 0].sum()
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    return iou

def similarity_compute(img_1: np.array, img_2: np.array):
    # img_1: np array (64x64x1)
    # img_2: np array (64x64x1)
    score = structural_similarity(img_1.squeeze()/255, img_2.squeeze()/255, data_range=1.0)
    return score

def score_compute(mask_list, img_2_list, img_1_list, img_0_list, balance_fac=0.25, weight=[0.4,0.4,0.2]):
    mask, ref_mask = mask_list
    img_2, ref_img_2 = img_2_list
    img_1, ref_img_1 = img_1_list
    img_0, ref_img_0 = img_0_list

    iou = iou_compute(ref_mask, mask)
    ssim_2 = similarity_compute( ref_img_2, img_2 )*weight[2]
    ssim_1 = similarity_compute( ref_img_1, img_1 )*weight[1]
    ssim_0 = similarity_compute( ref_img_0, img_0 )*weight[0]
    score = iou + balance_fac*(ssim_2 + ssim_1 + ssim_0)

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

import tensorflow as tf
import tensorflow_hub as hub
import requests
from typing import Generator, Iterable, List, Optional
import mediapy as media
from mediapy import set_show_save_dir
from PIL import Image

def gen_frame(frame_1, frame_2, model):
    time = np.array([0.5], dtype=np.float32)
    input_ = {'time': np.expand_dims(time, axis=0), 'x0': np.expand_dims(frame_1, axis=0),'x1': np.expand_dims(frame_2, axis=0)}
    mid_frame = model(input_)
    return mid_frame['image'][0].numpy()

def interpolate(gif, model):
    new_gif = []
    new_gif.append(gif[0])
    for i in range(1,len(gif)):
        new_frame = gen_frame(gif[i-1], gif[i], model)
        new_gif.append(new_frame)
        new_gif.append(gif[i])
    return new_gif

def interpolate_idx(gif, model, idx):
    new_gif = []
    new_gif.append(gif[0])
    for i in range(1,len(gif)):
        if i in idx:
            new_frame = gen_frame(gif[i-1], gif[i], model)
            new_gif.append(new_frame)
        new_gif.append(gif[i])
    return new_gif

def generate_gif(reference_mask, reference_img_2, reference_img_1, reference_img_0, filename):
    set_show_save_dir('./')
    model = hub.load("https://tfhub.dev/google/film/1")
    _UINT8_MAX_F = float(np.iinfo(np.uint8).max)
    
    gif = []
    for i in range(6):
        merged_img = merge(reference_img_2[i], reference_img_1[i], reference_img_0[i], reference_mask[i]).astype(np.float32) / _UINT8_MAX_F
        gif.append(merged_img)

    gif = interpolate_idx(gif, model, [2,3,4]) # 6 frames
    gif = interpolate_idx(gif, model, [3,4,5,6]) # 9 frames
    gif = interpolate(gif, model) # 17 frames
    gif = gif[len(gif)//5:]
    gif = interpolate(gif, model) # 65 frames
    gif = interpolate_idx(gif, model, [i for i in range(int(len(gif)//5*4), len(gif))]) # 65 frames
    gif = interpolate_idx(gif, model, [i for i in range(int(len(gif)//5*4), len(gif))]) # 65 frames
    
    media.show_images(gif)
    media.show_video(gif, fps=60, title=filename, codec='gif')