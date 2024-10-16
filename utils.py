import numpy as np
from skimage.metrics import structural_similarity
from cellpose import utils
import cv2
import imutils
from PIL import Image

def merge(img_2, img_1, img_0, mask):
    # img_2: (64,64,1) is the base 
    # img_1: (64,64,1) takes the R channel
    # img_0: (64,64,1) takes the G channel
    # mask: (64,64,1) is a segmentation mask
    # ---------
    img_2 = img_2.squeeze()*mask
    img_1 = img_1.squeeze()*mask
    img_0 = img_0.squeeze()*mask

    img_2 = np.expand_dims(img_2, axis=0)
    img_1 = np.expand_dims(img_1, axis=0)
    img_0 = np.expand_dims(img_0, axis=0)

    img_2_ = np.vstack([img_2, img_2, img_2])
    img_1_ = np.vstack([img_1, np.zeros(img_1.shape), np.zeros(img_1.shape)])
    img_0_ = np.vstack([np.zeros(img_0.shape), img_0, np.zeros(img_0.shape)])
    return np.dstack(img_2_*0.1 + img_1_*0.5 + img_0_*0.4).astype(int)

def iou_compute(outputs: np.array, labels: np.array):
    # source: https://www.kaggle.com/code/iezepov/fast-iou-scoring-metric-in-pytorch-and-numpy
    SMOOTH = 1e-6
    intersection = outputs * labels
    intersection = intersection[intersection != 0].sum()
    union = outputs + labels
    union = union[union != 0].sum()
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    return iou

def angle_sim_compute(mask_1: np.array, mask_2: np.array):
    # mask_1: np array (64x64)
    # mask_2: np array (64x64)
    try:
        outlines = utils.outlines_list(mask_1)
        ellipse = cv2.fitEllipse(np.array(outlines))
        _, _, angle_1 = ellipse

        outlines = utils.outlines_list(mask_2)
        ellipse = cv2.fitEllipse(np.array(outlines))
        _, _, angle_2 = ellipse
        return - np.abs(angle_1-angle_2) / 360
    except:
        print("Cannot compute angle!")
        return - 1000

def similarity_compute(img_1: np.array, img_2: np.array):
    # img_1: np array (64x64x1)
    # img_2: np array (64x64x1)
    score = structural_similarity(img_1.squeeze()/255, img_2.squeeze()/255, data_range=1.0)
    return score

def score_compute(mask_list_2, mask_list_1, img_2_list, img_1_list, img_0_list, balance_fac_iou=0.1, balance_fac_angle=0.3, balance_fac=0.25, weight=[0.4,0.4,0.2]):
    mask_2, ref_mask_2 = mask_list_2
    mask_1, ref_mask_1 = mask_list_1
    img_2, ref_img_2 = img_2_list
    img_1, ref_img_1 = img_1_list
    img_0, ref_img_0 = img_0_list

    iou_2 = iou_compute(ref_mask_2, mask_2)
    iou_1 = iou_compute(ref_mask_1, mask_1)
    angle_1 = angle_sim_compute(ref_mask_1, mask_1)
    angle_2 = angle_sim_compute(ref_mask_2, mask_2)
    ssim_2 = similarity_compute( ref_img_2, img_2 )*weight[2]
    ssim_1 = similarity_compute( ref_img_1, img_1 )*weight[1]
    ssim_0 = similarity_compute( ref_img_0, img_0 )*weight[0]
    score = iou_2 + \
            balance_fac_iou * iou_1 + \
            balance_fac_angle * angle_1 + \
            balance_fac * (ssim_2 + ssim_1 + ssim_0)

    return score

def determine_center(seg, size=64):
    # seg: (numpy) a map with indeces
    max_idx = np.amax(seg)
    shift_list = []
    mask = np.zeros((size,size))
    min_shift = 10000
    for i in range(1,int(max_idx)+1):
        z = np.zeros((size,size))
        z[seg == i] = 1
        
        shift = 0
        # https://stackoverflow.com/questions/51716954/how-to-center-the-nonzero-values-within-2d-numpy-array
        for k in range(2):
            nonempty = np.nonzero(np.any(z, axis=1-k))[0]
            first, last = nonempty.min(), nonempty.max()
            shift += np.abs(z.shape[k] - first - last)//2
        shift_list.append([z, shift])
    if len(shift_list) == 0:
        shift_list.append([np.zeros((size,size)), 0])
    return shift_list

import tensorflow as tf
import tensorflow_hub as hub
import requests
from typing import Generator, Iterable, List, Optional
import mediapy as media
from mediapy import set_show_save_dir
from PIL import Image

def visualize(data, filename, mode='L'):
    img = Image.fromarray(np.uint8(data), mode)
    img.save(filename)

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

import scipy.misc
from skimage.draw import line_aa
def add_outline(img, mask, channels=3):
    # img: 64x64x3 or 1
    # mask: 64x64
    # return: 64x64x3 or 1
    outlines = utils.outlines_list(mask)
    for o in outlines:
        for i in range(o.shape[0]):
            next_i = 0 if i+1 == o.shape[0] else i+1
            rr, cc, val = line_aa(o[i,1], o[i,0], o[next_i,1], o[next_i,0]) # o[:,0]: x points, o[:,1]: y points
            if channels==3:
                img[rr, cc, 2] = val * 50
                img[rr, cc, 1] = val * 50
                img[rr, cc, 0] = val * 50
            elif channels==1:
                img[rr, cc, 0] = val * 50
            else:
                img[rr, cc] = val * 50
    return img

def generate_gif(reference_mask_2, reference_mask_1, reference_mask_0, 
                 reference_img_2, reference_img_1, reference_img_0, 
                 filename, rotate_angle=0, flip_img=False):
    set_show_save_dir('./')
    model = hub.load("https://tfhub.dev/google/film/1")
    _UINT8_MAX_F = float(np.iinfo(np.uint8).max)
    
    gif = []
    for i in range(6):
        reference_mask = reference_mask_2[i] + reference_mask_1[i] + reference_mask_0[i]
        reference_mask[reference_mask != 0] = 1
        merged_img = merge(reference_img_2[i], reference_img_1[i], reference_img_0[i], reference_mask).astype(np.uint8)
        
        # if add_line:
        #     merged_img = add_outline(merged_img, reference_mask[i])

        # post-processing
        merged_img = rotate(merged_img, angle=rotate_angle)
        if flip_img:
            merged_img = flip(merged_img)
        gif.append(merged_img.astype(np.float32) / _UINT8_MAX_F)

    gif = interpolate_idx(gif, model, [2,3,4])
    gif = interpolate_idx(gif, model, [3,4,5,6])
    gif = interpolate(gif, model)
    gif = gif[len(gif)//4:]
    gif = interpolate_idx(gif, model, [i for i in range(int(len(gif)//5*4), len(gif))])
    gif = interpolate_idx(gif, model, [i for i in range(int(len(gif)//5*4), len(gif))])
    gif = interpolate(gif, model)

    media.show_video(gif, fps=100, title=filename, codec='gif', border=True)

# https://www.geeksforgeeks.org/how-to-rotate-an-image-using-python/
def rotate(img, angle=180):
    # img: 64x64x3
    rotated_image = imutils.rotate(img, angle=angle)
    return rotated_image

def flip(img):
    # img: 64x64x3
    img = Image.fromarray(img)
    flipped_img = np.asarray(img.transpose(method=Image.FLIP_LEFT_RIGHT))
    return flipped_img

def sharpen(img, channel=1, intensity=5, smoothness=0.5):
    # img: 64x64x3
    # Create a sharpening kernel
    kernel = np.array([[0, -smoothness, 0], [-smoothness, intensity, -smoothness], [0, -smoothness, 0]])
    # Apply the sharpening filter
    sharpened_image = cv2.filter2D(img[:,:,channel], -1, kernel)
    img[:,:,channel] = sharpened_image 
    return img
    