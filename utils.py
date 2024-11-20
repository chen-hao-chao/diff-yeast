import numpy as np
from skimage.metrics import structural_similarity
from cellpose import utils
import cv2
import imutils
from PIL import Image
import os

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

# def angle_compute(mask):
#     outlines = utils.outlines_list(mask)
#     ellipse = cv2.fitEllipse(np.array(outlines))
#     _, _, angle = ellipse
#     return angle

from scipy import stats as scistats
from sklearn import decomposition as skdecomp
def angle_compute(mask):
    y, x = np.where(mask)
    xy = np.hstack([x.reshape(-1, 1), y.reshape(-1, 1)])
    pca = skdecomp.PCA(n_components=2)
    pca = pca.fit(xy)
    eigenvecs = pca.components_
    # Calculate angle with arctan2
    angle = 180.0 * np.arctan2(eigenvecs[0][1], eigenvecs[0][0]) / np.pi
    # Rotate x coordinates
    x_rot = (x - x.mean()) * np.cos(np.pi * angle / 180) + (y - y.mean()) * np.sin(
        np.pi * angle / 180
    )
    # Check the skewness of the rotated x coordinate
    xsk = scistats.skew(x_rot)
    if xsk < 0.0:
        angle += 180
    # Map all angles to anti-clockwise
    angle = angle % 360
    return angle

def angle_sim_compute(mask_1: np.array, mask_2: np.array):
    # mask_1: np array (64x64)
    # mask_2: np array (64x64)
    try:
        angle_1 = angle_compute(mask_1)
        angle_2 = angle_compute(mask_2)

        return - np.abs(angle_1-angle_2) / 360
    except:
        print("Cannot compute angle!")
        return - 1000

def similarity_compute(img_1: np.array, img_2: np.array):
    # img_1: np array (64x64x1)
    # img_2: np array (64x64x1)
    score = structural_similarity(img_1.squeeze()/255, img_2.squeeze()/255, data_range=1.0)
    return score

def score_compute(mask_list_2, mask_list_1, mask_list_0, img_2_list, img_1_list, img_0_list, balance_fac=0.25, balance_fac_cat=0.1, weight=[0.4,0.4,0.2], weight_iou=[0.0,0.0,1.0]):
    mask_2, ref_mask_2 = mask_list_2
    mask_1, ref_mask_1 = mask_list_1
    mask_0, ref_mask_0 = mask_list_0
    img_2, ref_img_2 = img_2_list
    img_1, ref_img_1 = img_1_list
    img_0, ref_img_0 = img_0_list

    iou_2 = iou_compute(ref_mask_2, mask_2)*weight_iou[2]
    iou_1 = iou_compute(ref_mask_1, mask_1)*weight_iou[1]
    iou_0 = iou_compute(ref_mask_0, mask_0)*weight_iou[0]
    ssim_2 = similarity_compute( ref_img_2*ref_mask_2, img_2*mask_2 )*weight[2]
    ssim_1 = similarity_compute( ref_img_1*ref_mask_1, img_1*mask_1 )*weight[1]
    ssim_0 = similarity_compute( ref_img_0*ref_mask_0, img_0*mask_0 )*weight[0]

    category_ref = find_centres(ref_mask_2, ref_mask_1, ref_mask_0)
    category = find_centres(mask_2, mask_1, mask_0)
    category_score = -np.abs(category_ref - category)

    score = (iou_2 + iou_1 + iou_0) + balance_fac * (ssim_2 + ssim_1 + ssim_0) + balance_fac_cat * category_score
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

def interpolate(gif, tf_id, model):
    new_gif = []
    shift = [0 for i in range(len(tf_id))]
    new_gif.append(gif[0])
    for i in range(1,len(gif)):
        new_frame = gen_frame(gif[i-1], gif[i], model)
        new_gif.append(new_frame)
        for k in range(len(tf_id)):
            if tf_id[k] >= i:
                shift[k] = shift[k] + 1
        new_gif.append(gif[i])
    for i in range(len(shift)):
        tf_id[i] = tf_id[i] + shift[i]
    return new_gif, tf_id

def interpolate_idx(gif, tf_id, model, idx):
    new_gif = []
    shift = [0 for i in range(len(tf_id))]
    new_gif.append(gif[0])
    for i in range(1,len(gif)):
        if i in idx:
            new_frame = gen_frame(gif[i-1], gif[i], model)
            new_gif.append(new_frame)
            for k in range(len(tf_id)):
                if tf_id[k] >= i:
                    shift[k] = shift[k] + 1
        new_gif.append(gif[i])
    for i in range(len(shift)):
        tf_id[i] = tf_id[i] + shift[i]
    return new_gif, tf_id

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

def aggregate_masks(mask1, mask2, mask3):
    mask = mask3 + mask2 + mask1
    mask[mask != 0] = 1
    return mask

def generate_gif(reference_mask_2, reference_mask_1, reference_mask_0, 
                 reference_img_2, reference_img_1, reference_img_0, 
                 filename, rotate_angle=0, flip_img=False):
    set_show_save_dir('./')
    model = hub.load("https://tfhub.dev/google/film/1")
    _UINT8_MAX_F = float(np.iinfo(np.uint8).max)
    
    gif = []
    tf_id = []
    for i in range(len(reference_img_2)):
        reference_mask = aggregate_masks(reference_mask_2[i], reference_mask_1[i], reference_mask_0[i])
        merged_img = merge(reference_img_2[i], reference_img_1[i], reference_img_0[i], reference_mask).astype(np.uint8)
        
        # if add_line:
        #     merged_img = add_outline(merged_img, reference_mask[i])

        # post-processing
        merged_img = rotate(merged_img, angle=rotate_angle)
        if flip_img:
            merged_img = flip(merged_img)
        gif.append(merged_img.astype(np.float32) / _UINT8_MAX_F)
        tf_id.append(i)

    gif, tf_id = interpolate_idx(gif, tf_id, model, [2,3,4])
    gif, tf_id = interpolate_idx(gif, tf_id, model, [3,4,5,6])
    gif, tf_id = interpolate(gif, tf_id, model)
    for j in range(len(tf_id)):
        tf_id[j] = tf_id[j] - int(len(gif)//4)
    gif = gif[len(gif)//4:]
    gif, tf_id = interpolate_idx(gif, tf_id, model, [i for i in range(int(len(gif)//5*4), len(gif))])
    gif, tf_id = interpolate_idx(gif, tf_id, model, [i for i in range(int(len(gif)//5*4), len(gif))])
    gif, tf_id = interpolate(gif, tf_id, model)
    gif, tf_id = interpolate(gif, tf_id, model)
    print("Length: ", len(gif), len(tf_id))

    media.show_video(gif, fps=100, title=filename, codec='gif', border=True)
    np.savetxt(filename+'.txt', tf_id, fmt='%d')
    # write_video(gif, filename+'_video.mp4')

# def write_video(gif, filename):
#     out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 5, (64,64))
#     for i in range(len(gif)):
#         img = np.copy(gif[i])
#         img[:,:,0] = gif[i][:,:,2] # 1 <- 2
#         img[:,:,2] = gif[i][:,:,0]
#         out.write(img)
#     out.release()

def stack_imgs(img1, img2, img3):
    # output: 3x64x64
    img1 = np.expand_dims(img1, axis=0)
    img2 = np.expand_dims(img2, axis=0)
    img3 = np.expand_dims(img3, axis=0)
    img = np.vstack([img1, img2, img3])
    return img

def flip_to_normalize(img1, img2, img3, mask1, mask2, mask3):
    flipped_mask1 = flip(mask1)
    flipped_mask2 = flip(mask2)
    flipped_mask3 = flip(mask3)

    flipped_img1 = flip(img1.squeeze())
    flipped_img2 = flip(img2.squeeze())
    flipped_img3 = flip(img3.squeeze())

    return flipped_img1, flipped_img2, flipped_img3, flipped_mask1, flipped_mask2, flipped_mask3

def rotate_to_normalize(img1, img2, img3, mask1, mask2, mask3):
    mask = aggregate_masks(mask1, mask2, mask3)
    angle = angle_compute(mask)

    # rotate images and masks
    rotated_mask1 = rotate(mask1, angle=angle)
    rotated_mask2 = rotate(mask2, angle=angle)
    rotated_mask3 = rotate(mask3, angle=angle)

    rotated_img1 = rotate(img1.squeeze(), angle=angle)
    rotated_img2 = rotate(img2.squeeze(), angle=angle)
    rotated_img3 = rotate(img3.squeeze(), angle=angle)

    return rotated_img1, rotated_img2, rotated_img3, rotated_mask1, rotated_mask2, rotated_mask3

from skimage import transform as sktrans
# https://www.geeksforgeeks.org/how-to-rotate-an-image-using-python/
def rotate(img, angle=180):
    # img: 64x64x3
    # rotated_image = imutils.rotate(img, angle=angle)
    rotated_image = sktrans.rotate( image=img,angle=angle,resize=False,preserve_range=True)
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

# Sophie's code
def get_mother_daughter(centroids0_x, centroids1_x, cluster_size0, cluster_size1):
    # (MotherCell: L; DaughterCell: R) -> 1
    # (MotherCell: R; DaughterCell: L) -> 0
    if centroids0_x < centroids1_x:
        if cluster_size0 > cluster_size1:
            category = 1
        else:
            category = 0
    else:
        if cluster_size0 > cluster_size1:
            category = 0
        else:
            category = 1
    return category

from sklearn.cluster import KMeans
def find_centres(mask1, mask2, mask3, num_clusters=2):
    mask = aggregate_masks(mask1, mask2, mask3)
    coords = np.argwhere(mask)
    # Apply KMeans clustering to identify 'num_clusters' clusters
    kmeans = KMeans(n_clusters=2, random_state=0, n_init='auto')
    kmeans.fit(coords)

    # Get the cluster centers and labels
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    cluster_sizes = np.bincount(labels)

    # Convert centroids to integer pixel coordinates
    centroids_int = [(int(c[1]), int(c[0])) for c in centroids]  # Swap to (x, y) format

    return get_mother_daughter(centroids_int[0][0], centroids_int[1][0], cluster_sizes[0], cluster_sizes[1])