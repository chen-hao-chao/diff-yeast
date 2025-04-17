from dataloader import CellDataLoader
from torch.utils.data import DataLoader
import numpy as np
import os

root_dir_no_gfp='/fs01/datasets/yeast-imgs/cellcycle_single_cell_crops/128/select_proteins/no_GFP/R1'
df_no_gfp_sublist = [f for f in os.listdir(root_dir_no_gfp) if f.split('.')[1]=='tiff']

print(len(df_no_gfp_sublist))
print("constructing dataloader...")
dataset_no_gfp = CellDataLoader(root_dir=root_dir_no_gfp, data_list=df_no_gfp_sublist)
data_loader_no_gfp = DataLoader(dataset_no_gfp, batch_size=100, shuffle=False)

mean_value = 0
i = 0

print("start computation...")
for imgs in data_loader_no_gfp:
    print("Calculating the auto-flou. for {}/{} images...".format(i, len(data_loader_no_gfp)))
    imgs_channel_no_gfp_0 = imgs[:,0,:,:].numpy()
    imgs_channel_no_gfp_0 = imgs_channel_no_gfp_0
    acc_mean = np.mean(imgs_channel_no_gfp_0[imgs_channel_no_gfp_0 != 0])
    i += 1
    mean_value += acc_mean

print(mean_value / i)