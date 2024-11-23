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
        f = os.path.join(directory, filename, str(j)+'_structure.gif') # filename
        f_b = os.path.join(directory, filename, str(j)+'_background.gif') # filename

        if os.path.isfile(f) and os.path.isfile(f_b):
            tfid = np.loadtxt(os.path.join(directory_tfid, filename, str(j)+'_neu_structure.txt'), dtype=int)
            frames = iio.imread(f)
            # add the text
            i = 0
            true_frames = []
            for frame in frames:
                text = "T" if (i in tfid) else "F"
                foo = cv2.putText(frame, text,
                    (2,120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    .40,
                    (255, 255, 255)
                )
                if (i in tfid):
                    true_frames.append(foo)
                i = i+1
            kargs = { 'duration': 0.1 }
            iio.imwrite(os.path.join(output_directory, filename, str(j)+'_structure.gif'), frames, loop=0)
            iio.imwrite(os.path.join(output_directory, filename, str(j)+'_structure_true.gif'), true_frames, loop=0, fps=1)

            # -----
            frames_b = iio.imread(f_b)
            for frame in frames_b:
                text = "T" if (i in tfid) else "F"
                foo = cv2.putText(frame, text,
                    (2,120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    .40,
                    (255, 255, 255)
                )
                if (i in tfid):
                    true_frames.append(foo)
                i = i+1
            
            iio.imwrite(os.path.join(output_directory, filename, str(j)+'_background.gif'), frames, loop=0)
            iio.imwrite(os.path.join(output_directory, filename, str(j)+'_background_true.gif'), true_frames, loop=0, fps=1)