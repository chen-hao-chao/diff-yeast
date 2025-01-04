import imageio.v3 as iio
import os
from utils import add_outline, interpolate, visualize, sharpen
import numpy as np
import pdb
from utils import pseudo_segmentation

directory = 'test_fig'
output_directory = 'test_fig_outlined'
num_frames = 5
eps = 8 # hyper-parameter: 0 (loose) ~ 255 (tight). Determine the tightness of outline.
try:
    os.mkdir(output_directory) 
except:
    print("directory exists.")

for filename in os.listdir(directory):
    try:
        os.mkdir(os.path.join(output_directory, filename))
    except:
        print("directory exists.")
    for j in range(num_frames):
        f_b = os.path.join(directory, filename, str(j)+'_default.gif') # filename
        if os.path.isfile(f_b):
            frames = iio.imread(f_b)
            new_gif = []
            for i in range(frames.shape[0]):
                img = frames[i][:,:,0]+frames[i][:,:,1]+frames[i][:,:,2] / 3
                new_gif.append(np.clip((img - eps) / np.max(img), a_min=0, a_max=1))
            masks, _ = pseudo_segmentation(new_gif)
            
            new_gif = []
            for i in range(frames.shape[0]):
                new_frame = add_outline(frames[i], masks[i], channels=3)
                new_gif.append(new_frame)

            iio.imwrite(os.path.join(output_directory, filename, str(j)+'_default.gif'), new_gif, loop=0)