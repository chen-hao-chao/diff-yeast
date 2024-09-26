import imageio.v3 as iio
import cv2
import os
from utils import add_outline, interpolate, visualize
from cellpose import utils, denoise, io
import numpy as np
import tensorflow_hub as hub

model = denoise.CellposeDenoiseModel(gpu=True, model_type="cyto3",restore_type="denoise_cyto3")
model_inter = hub.load("https://tfhub.dev/google/film/1")

directory = 'gif_verify'
output_directory = 'gif_verify_seg'
try:
    os.mkdir(output_directory) 
except:
    print("directory exists.")


for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isfile(f):
        frames = iio.imread(f)
        reference_img = []
        frame_list = []
        
        for i in range(frames.shape[0]):
            mix = frames[i][:,:,2]*5+frames[i][:,:,1]+frames[i][:,:,0]
            reference_img.append(mix)
            frame_list.append(frames[i]) # 64x64x3
        masks, _, _, imgs_dn = model.eval(reference_img, channels=[0,0])
        new_gif = []
        for i in range(frames.shape[0]):
            mask = np.zeros((64,64))
            mask[masks[i] != 0] = 1
            new_frame = add_outline(frame_list[i], mask, channels=3)
            new_gif.append(new_frame)

        # new_gif = interpolate(new_gif, model_inter)
        # iio.imwrite("test.gif", new_gif, loop=0)
        iio.imwrite(os.path.join(output_directory, filename), new_gif, loop=0)