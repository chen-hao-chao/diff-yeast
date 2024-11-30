import imageio.v3 as iio
import cv2
import os
from utils import add_outline, interpolate, visualize, sharpen
from cellpose import utils, denoise, io
import numpy as np
import tensorflow_hub as hub

model = denoise.CellposeDenoiseModel(gpu=True, model_type="cyto3",restore_type="denoise_cyto3")
model_inter = hub.load("https://tfhub.dev/google/film/1")

directory = 'test_fig'
output_directory = 'test_fig_sharpened'
num_frames = 1
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
        f = os.path.join(directory, filename, str(j)+'_structure.gif') # filename
        f_b = os.path.join(directory, filename, str(j)+'_default.gif') # filename
        if os.path.isfile(f) and os.path.isfile(f_b):
            frames = iio.imread(f)
            new_gif = []
            ori_gif = []
            
            for i in range(frames.shape[0]):
                new_frame = sharpen(frames[i], channel=1, intensity=3.5, smoothness=0.5)
                new_gif.append(new_frame)
                ori_gif.append(frames[i])

            iio.imwrite(os.path.join(output_directory, filename, str(j)+'_structure.gif'), new_gif, loop=0)

            # ----

            frames = iio.imread(f_b)
            new_gif = []
            ori_gif = []
            
            for i in range(frames.shape[0]):
                new_frame = sharpen(frames[i], channel=2, intensity=3.5, smoothness=0.5)
                new_gif.append(new_frame)
                ori_gif.append(frames[i])

            iio.imwrite(os.path.join(output_directory, filename, str(j)+'_default.gif'), new_gif, loop=0)