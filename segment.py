import numpy as np
import time, os, sys
from urllib.parse import urlparse
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200
from cellpose import utils, denoise, io
import tifffile as tiff

filename = "/datasets/yeast-imgs/R1/Rep1_Plate9_016024010_cell_3085833.tiff"
imgs = [tiff.imread(filename)[2,:,:], tiff.imread(filename)[1,:,:], tiff.imread(filename)[0,:,:]]

io.logger_setup()
model = denoise.CellposeDenoiseModel(gpu=True, model_type="cyto3",restore_type="denoise_cyto3")
masks, flows, styles, imgs_dn = model.eval(imgs, channels=[0,0])

z = np.zeros((64,64))
seg = masks[0].squeeze()
z[seg == 3] = 1

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
plt.savefig('seg.png')