import numpy as np
import time, os, sys
from urllib.parse import urlparse
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200
from cellpose import utils, denoise, io
import tifffile as tiff
from utils import determine_center, merge, visualize

size = 128
# mode = "colored"
# mode = "channel"
data = "Rep1_Plate10_001003004_cell_3088799"
split_stage = 4

filename = "/datasets/yeast-imgs/cellcycle_single_cell_crops_128/R1/"+data+".tiff"
imgs = [tiff.imread(filename)[2,:,:], tiff.imread(filename)[1,:,:], tiff.imread(filename)[0,:,:]]

io.logger_setup()
model = denoise.CellposeDenoiseModel(gpu=True, model_type="cyto3",restore_type="denoise_cyto3")
masks, flows, styles, imgs_dn = model.eval(imgs, channels=[0,0])

shift_list = sorted(determine_center(masks[0].squeeze(), size=size), key=lambda x: x[1])
# z = shift_list[0][0]
z = shift_list[0][0] if (split_stage < 2 or len(shift_list)==1) else np.clip(shift_list[0][0] + shift_list[1][0], 0, 1)

# if mode == "colored":
merged_img = merge((imgs_dn[2] * 255 / np.amax(imgs_dn[2])).astype(int), (imgs_dn[1] * 255 / np.amax(imgs_dn[1])).astype(int), (imgs_dn[0] * 255 / np.amax(imgs_dn[0])).astype(int), z)
visualize(merged_img, data+"_colored.png", mode="RGB")

# else:
plt.subplot(2,3,4)
plt.imshow(imgs_dn[0].squeeze()*z, cmap="gray", vmin=0, vmax=1)
plt.axis('off')
plt.title("seg - channel 3")

plt.subplot(2,3,5)
plt.imshow(imgs_dn[1].squeeze()*z, cmap="gray", vmin=0, vmax=1)
plt.axis('off')
plt.title("seg - channel 2")

plt.subplot(2,3,6)
plt.imshow(imgs_dn[2].squeeze()*z, cmap="gray", vmin=0, vmax=1)
plt.axis('off')
plt.title("seg - channel 1")


plt.subplot(2,3,1)
plt.imshow(imgs_dn[0].squeeze(), cmap="gray", vmin=0, vmax=1)
plt.axis('off')
plt.title("img - channel 3")

plt.subplot(2,3,2)
plt.imshow(imgs_dn[1].squeeze(), cmap="gray", vmin=0, vmax=1)
plt.axis('off')
plt.title("img - channel 2")

plt.subplot(2,3,3)
plt.imshow(imgs_dn[2].squeeze(), cmap="gray", vmin=0, vmax=1)
plt.axis('off')
plt.title("img - channel 1")

plt.tight_layout()
plt.savefig(data+"_seg.png")