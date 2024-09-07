from PIL import Image
import numpy

im = Image.open('/datasets/yeast-imgs/R1/Rep1_Plate4_008016002_cell_1100167.tiff')
imarray = numpy.array(im)
print(imarray.shape)
print(imarray)

jpeg_image = im.convert("RGB")

# Save the JPEG image
jpeg_image.save("example.jpg")