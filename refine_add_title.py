import imageio.v3 as iio
import cv2
import os

directory = 'test_fig_sharpened'
output_directory = 'test_fig_sharpened_title'
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
            frames = iio.imread(f)
            # add the text
            for frame in frames:
                foo = cv2.putText(frame, filename.split(".")[0],
                    (12,10), #(6,10), #(12, 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    .28,
                    (255, 255, 255)
                )
            iio.imwrite(os.path.join(output_directory, filename, str(j)+'.gif'), frames, loop=0)