import imageio.v3 as iio
import cv2
import os
import numpy as np

directory_tfid = 'test_fig'
directory = 'test_fig_sharpened_title'
output_directory = 'test_fig_sharpened_title_tfid'
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
        f = os.path.join(directory, filename, str(j)+'.gif')
        if os.path.isfile(f):
            tfid = np.loadtxt(os.path.join(directory_tfid, filename, str(j)+'.txt'), dtype=int)
            frames = iio.imread(f)
            # add the text
            i = 0
            for frame in frames:
                text = "T" if (i in tfid) else "F"
                foo = cv2.putText(frame, text,
                    (2,60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    .28,
                    (255, 255, 255)
                )
                i = i+1
            iio.imwrite(os.path.join(output_directory, filename, str(j)+'.gif'), frames, loop=0)