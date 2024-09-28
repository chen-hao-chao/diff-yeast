import imageio.v3 as iio
import cv2
import os
from utils import add_outline, interpolate, visualize, sharpen
from cellpose import utils, denoise, io
import numpy as np
import tensorflow_hub as hub

model = denoise.CellposeDenoiseModel(gpu=True, model_type="cyto3",restore_type="denoise_cyto3")
model_inter = hub.load("https://tfhub.dev/google/film/1")

directory = 'gif_verify'
output_directory = 'gif_verify_sharpened'
try:
    os.mkdir(output_directory) 
except:
    print("directory exists.")


for filename in os.listdir(directory):
    f = os.path.join(directory, filename) # filename
    if os.path.isfile(f):
        frames = iio.imread(f)
        new_gif = []
        ori_gif = []
        
        for i in range(frames.shape[0]):
            new_frame = sharpen(frames[i], channel=1, intensity=3.5, smoothness=0.5)
            new_gif.append(new_frame)
            ori_gif.append(frames[i])

        iio.imwrite(os.path.join(output_directory, filename), new_gif, loop=0)