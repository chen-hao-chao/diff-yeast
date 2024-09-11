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
            shift += np.abs(z.shape[k] - first - last)//2 #np.abs(z.shape[k]//2 - (last - first)//2)
        if shift < min_shift:
            mask = z
            min_shift = shift
    return mask

total = 100
bs = 100
for i in range(6):
    loaded_df = pd.read_csv('/datasets/yeast-imgs/single_cell_annotations/group_by_cycle_rep1_filename_dict.csv', index_col='correctedMaxCycle_num')
    filenames = loaded_df['filename'].apply(literal_eval)
    result_dict = filenames[i][:total]

    dataset = MyData(root_dir='/datasets/yeast-imgs/R1', data_list=result_dict, phase=i)
    data_loader = DataLoader(dataset, batch_size=bs, shuffle=True)
    
    # setup models
    io.logger_setup()
    model = denoise.CellposeDenoiseModel(gpu=True, model_type="cyto3",restore_type="denoise_cyto3")

    img_list_2 = []
    img_list_1 = []
    img_list_0 = []
    for imgs, labels in data_loader:
        # torch.Size([100, 3, 64, 64])
        imgs_channel_2 = imgs[:,2,:,:]
        imgs_channel_1 = imgs[:,1,:,:]
        imgs_channel_0 = imgs[:,0,:,:]
        for k in range(bs):
            img_list_2.append(imgs_channel_2[k,:,:])
            img_list_1.append(imgs_channel_1[k,:,:])
            img_list_0.append(imgs_channel_0[k,:,:])

    masks_2, _, _, imgs_dn_2 = model.eval(img_list_2, channels=[0,0])
    _, _, _, imgs_dn_1 = model.eval(img_list_1, channels=[0,0])
    _, _, _, imgs_dn_0 = model.eval(img_list_0, channels=[0,0])

    
    # output code
    for j in range(bs):
        print(str(j)+"-th image...")
        mask = determine_center(masks_2[j].squeeze())

        plt.subplot(2,3,4)
        plt.imshow(imgs_dn_2[j].squeeze()*mask, cmap="gray", vmin=0, vmax=1)
        plt.axis('off')
        plt.title("seg - channel 3")

        plt.subplot(2,3,5)
        plt.imshow(imgs_dn_1[j].squeeze()*mask, cmap="gray", vmin=0, vmax=1)
        plt.axis('off')
        plt.title("seg - channel 2")

        plt.subplot(2,3,6)
        plt.imshow(imgs_dn_0[j].squeeze()*mask, cmap="gray", vmin=0, vmax=1)
        plt.axis('off')
        plt.title("seg - channel 1")


        plt.subplot(2,3,1)
        plt.imshow(imgs_dn_2[j].squeeze(), cmap="gray", vmin=0, vmax=1)
        plt.axis('off')
        plt.title("img - channel 3")

        plt.subplot(2,3,2)
        plt.imshow(imgs_dn_1[j].squeeze(), cmap="gray", vmin=0, vmax=1)
        plt.axis('off')
        plt.title("img - channel 2")

        plt.subplot(2,3,3)
        plt.imshow(imgs_dn_0[j].squeeze(), cmap="gray", vmin=0, vmax=1)
        plt.axis('off')
        plt.title("img - channel 1")

        plt.tight_layout()
        plt.savefig('results/'+str(i+1)+'/'+str(j+1)+'.png')