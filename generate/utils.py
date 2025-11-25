import numpy as np
from skimage.metrics import structural_similarity
from cellpose import utils
import cv2
from PIL import Image
import os

from skimage.draw import line_aa
import scipy as sp

from scipy import stats as scistats
from sklearn import decomposition as skdecomp
import time
import torch
from PIL import Image

import pdb

def pseudo_segmentation(img_list):
    masks = []
    for i in range(len(img_list)):
        mask = np.zeros(img_list[i].shape)
        mask[img_list[i] != 0] = 1
        masks.append(mask)
    return masks, img_list

def shift_intensity(img, value=0):
    img = img - value 
    img = np.clip(img, a_min=0, a_max=None)
    return img

def normalize_intensity(img, avg=0.2):
    # Normalize the intensity to avg.
    # Set avg=0 to deactivate the avgerage shifting operation.
    img = img / np.amax(img)
    if avg != 0:
        img = np.clip(img + (avg-np.mean(img)), 0, 1)
    return (img * 255).astype(np.uint8)

# Source: https://stackoverflow.com/questions/33548639/how-can-i-smooth-elements-of-a-two-dimensional-array-with-differing-gaussian-fun
def smooth_mask(mask, sigma=1):
    mask = sp.ndimage.filters.gaussian_filter(mask, sigma, mode='constant')
    return mask

def merge(img_2, img_1, img_0, mask, apply_mask=True, mode='default'):
    # img_2: (64,64,1) is the base 
    # img_1: (64,64,1) takes the R channel
    # img_0: (64,64,1) takes the G channel
    # mask: (64,64,1) is a segmentation mask
    # ---------
    mask = smooth_mask(mask, sigma=3)
    img_2 = img_2.squeeze()*mask if apply_mask else img_2.squeeze()
    img_1 = img_1.squeeze()*mask if apply_mask else img_1.squeeze()
    img_0 = img_0.squeeze()*mask if apply_mask else img_0.squeeze()

    img_2 = np.expand_dims(img_2, axis=0)
    img_1 = np.expand_dims(img_1, axis=0)
    img_0 = np.expand_dims(img_0, axis=0)

    if mode == 'default':
        img_2_ = np.vstack([img_2, img_2, img_2])
        img_1_ = np.vstack([img_1, np.zeros(img_1.shape), np.zeros(img_1.shape)])
        img_0_ = np.vstack([np.zeros(img_0.shape), img_0, np.zeros(img_0.shape)])
        return np.dstack(img_2_*0.1 + img_1_*0.5 + img_0_*0.4).astype(int)
    else:
        img_2_ = np.vstack([np.zeros(img_2.shape), np.zeros(img_2.shape), img_2])
        img_1_ = np.vstack([img_1, np.zeros(img_1.shape), np.zeros(img_1.shape)])
        img_0_ = np.vstack([np.zeros(img_0.shape), np.zeros(img_0.shape), np.zeros(img_0.shape)])
        return np.dstack(img_2_*0.5 + img_1_*0.5 + img_0_*0.0).astype(int)

def iou_compute(outputs: np.array, labels: np.array):
    # source: https://www.kaggle.com/code/iezepov/fast-iou-scoring-metric-in-pytorch-and-numpy
    SMOOTH = 1e-6
    intersection = outputs * labels
    intersection = intersection[intersection != 0].sum()
    union = outputs + labels
    union = union[union != 0].sum()
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    return iou


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

def score_compute(mask_list_2, mask_list_1, mask_list_0, img_2_list, img_1_list, img_0_list, balance_fac=0.25, weight=[0.4,0.4,0.2], weight_iou=[0.0,0.0,1.0]):
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

    score = (iou_2 + iou_1 + iou_0) + balance_fac * (ssim_2 + ssim_1 + ssim_0)
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

def visualize(data, filename, mode='L'):
    # data should be within the range [0,255]
    img = Image.fromarray(np.uint8(data), mode)
    img.save(filename)

def gen_frame(frame_1, frame_2, model, 
              precision=torch.float16, device=torch.device('cuda')):
    frame_1 = torch.tensor(frame_1).to(precision).to(device)
    frame_2 = torch.tensor(frame_2).to(precision).to(device)
    time = frame_1.new_full((1, 1), .5)

    width, height, channel = frame_1.shape
    frame_1 = frame_1.view(1, width, height, channel).permute(0, 3, 1, 2)
    frame_2 = frame_2.view(1, width, height, channel).permute(0, 3, 1, 2)
    
    mid_frame = model(frame_1, frame_2, time)
    mid_frame = mid_frame.view(channel, width, height).permute(1, 2, 0)
    return mid_frame.detach().cpu().numpy()

def gen_frame_parallel(frame_1, frame_2, model, 
              precision=torch.float16, device=torch.device('cuda')):
    frame_1 = torch.tensor(frame_1).to(precision).to(device)
    frame_2 = torch.tensor(frame_2).to(precision).to(device)
    time = frame_1.new_full((frame_1.shape[0], 1), .5)

    bs, width, height, channel = frame_1.shape
    frame_1 = frame_1.view(bs, width, height, channel).permute(0, 3, 1, 2)
    frame_2 = frame_2.view(bs, width, height, channel).permute(0, 3, 1, 2)
    
    mid_frame = model(frame_1, frame_2, time)
    mid_frame = mid_frame.view(bs, channel, width, height).permute(0, 2, 3, 1)
    return mid_frame.detach().cpu().numpy()

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

def interpolate_parallel(gif, tf_id, model):
    new_gif = []
    shift = [0 for i in range(len(tf_id))]
    frame_1s = np.array(gif[:-1])
    frame_2s = np.array(gif[1:])
    new_frames = gen_frame_parallel(frame_1s, frame_2s, model)
    new_gif.append(gif[0])
    for i in range(1,len(gif)):
        new_frame = new_frames[i-1]
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

def save_real_frames(gif, filename, mode='visualized'):
    for i in range(len(gif)):
        if mode == "visualized":
            im = Image.fromarray(gif[i].astype(np.uint8))
            im.save(filename + f"_real_frame_{i}.png")
        else:
            img_np = np.array(gif[i].astype(np.uint8))
            most_common_value = np.bincount(img_np.flatten()).argmax()
            img_np[img_np < most_common_value] = most_common_value
            # save
            im = Image.fromarray(img_np, 'L')
            im.save(filename + f"_real_frame_{i}_{mode}.png")

def generate_gif(reference_mask_2, reference_mask_1, reference_mask_0, 
                 reference_img_2, reference_img_1, reference_img_0, 
                 filepath, filename, rotate_angle=0, flip_img=False, apply_mask=True, 
                 mode='default', reverse_playing=False, 
                 model_path='h/chchao/film_net_fp16.pt'):
    device = torch.device('cuda')
    precision = torch.float16
    model = torch.jit.load(model_path, map_location='cpu')
    model.eval().to(device=device, dtype=precision)
    _UINT8_MAX_F = float(np.iinfo(np.uint8).max)
    
    gif = []
    tf_id = []
    save_gif = []
    for i in range(len(reference_img_2)):
        reference_mask = aggregate_masks(reference_mask_2[i], reference_mask_1[i], reference_mask_0[i])
        merged_img = merge(reference_img_2[i], reference_img_1[i], reference_img_0[i], reference_mask, apply_mask=apply_mask, mode=mode).astype(np.uint8)
        
        # post-processing
        if flip_img:
            merged_img = flip(merged_img)
        if rotate_angle != 0:
            merged_img = rotate(merged_img, angle=rotate_angle)
        
        save_gif.append(merged_img)
        gif.append(merged_img.astype(np.float32) / _UINT8_MAX_F)
        tf_id.append(i)

    save_real_frames(save_gif, os.path.join(filepath, 'real_frames', filename))
    save_real_frames(reference_img_2, os.path.join(filepath, 'real_frames', filename), mode='base')
    save_real_frames(reference_img_1, os.path.join(filepath, 'real_frames', filename), mode='nucleus')
    save_real_frames(reference_img_0, os.path.join(filepath, 'real_frames', filename), mode='structure')

    start = time.time()
    gif, tf_id = interpolate_idx(gif, tf_id, model, [i for i in range(0,int(len(gif)//6*2))])
    gif, tf_id = interpolate_parallel(gif, tf_id, model)
    gif, tf_id = interpolate_parallel(gif, tf_id, model)
    gif, tf_id = interpolate_parallel(gif, tf_id, model)
    gif, tf_id = interpolate_parallel(gif, tf_id, model)
    gif, tf_id = interpolate_parallel(gif, tf_id, model)
    end = time.time()
    print("Time (Pure Interpolation): {}".format(end - start))
    print("Length: ", len(gif), len(tf_id))

    masks = []
    for i in range(len(gif)):
        mask0 = np.zeros(gif[i][:,:,0].shape)
        mask0[gif[i][:,:,0] != 0] = 1
        mask1 = np.zeros(gif[i][:,:,1].shape)
        mask1[gif[i][:,:,1] != 0] = 1
        mask2 = np.zeros(gif[i][:,:,2].shape)
        mask2[gif[i][:,:,2] != 0] = 1
        masks.append(aggregate_masks(mask0, mask1, mask2))

    # use this or
    # new_gif = []
    # for i in range(len(gif)):
    #     lined_img = add_outline(gif[i], masks[i], channels=3)
    #     new_gif.append(lined_img)
    # this (no outline)
    new_gif = gif

    gif_uint8 = [np.clip((frame * 255 if frame.max() <= 1 else frame), 0, 255).astype(np.uint8) for frame in new_gif]

    if reverse_playing:
        frames = [Image.fromarray(frame) for frame in gif_uint8[::-1]]
        frames[0].save(os.path.join(filepath, filename+'.gif'), save_all=True, append_images=frames[1:],
                        duration=10, loop=0, optimize=True)
        np.savetxt(os.path.join(filepath, filename+'.txt'), tf_id[::-1], fmt='%d')
    else:
        frames = [Image.fromarray(frame) for frame in gif_uint8]
        frames[0].save(os.path.join(filepath, filename+'.gif'), save_all=True, append_images=frames[1:],
                        duration=10, loop=0, optimize=True)
        np.savetxt(os.path.join(filepath, filename+'.txt'), tf_id, fmt='%d')

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