import imageio.v3 as iio
import cv2
import os
import pandas as pd
from ast import literal_eval

directory = 'test_fig_sharpened'
output_directory = 'test_fig_sharpened_title'
num_frames = 1
try:
    os.mkdir(output_directory) 
except:
    print("directory exists.")

dfs = pd.read_excel("ORF_structure_names.xlsx")

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
            # add the text
            for frame in frames:
                
                dfss = dfs[dfs['ORF']==filename.split(".")[0]]['Structure']
                dfss = '('+dfss[dfss.index.values.astype(int)[0]]+')'

                scale = 0.40
                face = cv2.FONT_HERSHEY_SIMPLEX
                textsize = cv2.getTextSize(filename.split(".")[0], face, scale, 2)[0]
                textsize_s = cv2.getTextSize(dfss, face, scale, 2)[0]
                textX = int((frame.shape[1] - textsize[0]) / 2)
                textY = 12
                textX_s = int((frame.shape[1] - textsize_s[0]) / 2)
                textY_s = 24
                

                foo = cv2.putText(frame, filename.split(".")[0],
                    (textX, textY),
                    face,
                    scale,
                    (255, 255, 255)
                )
                foo = cv2.putText(frame, dfss,
                    (textX_s, textY_s),
                    face,
                    scale,
                    (255, 255, 255)
                )
            iio.imwrite(os.path.join(output_directory, filename, str(j)+'_default.gif'), frames, loop=0)

            # ----

            frames_b = iio.imread(f_b)
            for frame in frames_b:
                foo = cv2.putText(frame, filename.split(".")[0],
                    (textX, textY), #(6,10), #(12, 10),
                    face,
                    scale,
                    (255, 255, 255)
                )
                foo = cv2.putText(frame, dfss,
                    (textX_s, textY_s),
                    face,
                    scale,
                    (255, 255, 255)
                )

            iio.imwrite(os.path.join(output_directory, filename, str(j)+'_background.gif'), frames_b, loop=0)