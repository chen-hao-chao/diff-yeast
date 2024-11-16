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
try:
    os.mkdir(output_directory) 
except:
    print("directory exists.")


for filename in os.listdir(directory):
    try:
        os.mkdir(os.path.join(output_directory, filename))
    except:
        print("directory exists.")
    for j in range(3):
        f = os.path.join(directory, filename, str(j)+'.gif') # filename
        if os.path.isfile(f):
            frames = iio.imread(f)
            new_gif = []
            ori_gif = []
            
            for i in range(frames.shape[0]):
                new_frame = sharpen(frames[i], channel=1, intensity=3.5, smoothness=0.5)
                new_gif.append(new_frame)
                ori_gif.append(frames[i])

            iio.imwrite(os.path.join(output_directory, filename, str(j)+'.gif'), new_gif, loop=0)
            # out = cv2.VideoWriter(os.path.join(output_directory, filename, str(j)+'.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 5, (64,64))
            # for i in range(len(new_gif)):
            #     img = np.copy(new_gif[i])
            #     img[:,:,0] = new_gif[i][:,:,2] # 1 <- 2
            #     img[:,:,2] = new_gif[i][:,:,0]
            #     out.write(img)
            # out.release()