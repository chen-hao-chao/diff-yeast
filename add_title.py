import imageio.v3 as iio
import cv2
import os

directory = 'gif_verify_seg'
output_directory = 'gif_verify_add_title'
try:
    os.mkdir(output_directory) 
except:
    print("directory exists.")

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isfile(f):
        frames = iio.imread(f)
        # add the text
        for frame in frames:
            foo = cv2.putText(frame, filename.split(".")[0],
                (12, 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                .28,
                (255, 255, 255)
            )
        iio.imwrite(os.path.join(output_directory, filename), frames, loop=0)