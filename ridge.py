import cv2
from skimage import data, img_as_ubyte, filters
from skimage import color
import skimage.io
from skimage.filters import meijering, sato, frangi, hessian, gaussian
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters.rank import median, enhance_contrast

from utils import warp_conveyer, take_strip, horizontal_gauss, detect_ridges


def original(image, **kwargs):
    """Return the original image, ignoring any kwargs."""
    return image


#image = color.rgb2gray(data.retina())[300:700, 700:900]
# load image from file using skimage
src = '/Users/matejnevlud/github/LN3/hsv_test/13.jpeg'

# load 9-13.jpeg
src = '/Users/matejnevlud/github/LN3/hsv_test/'
ridges_frames = []
for i in range(9, 15):
    image = skimage.io.imread(src + str(i) + ".jpeg", as_gray=True)
    image = warp_conveyer(image)
    #image = take_strip(image)
    image = horizontal_gauss(image)
    ridges = detect_ridges(image)
    # draw line in the middle of the image, use val 128
    ridges[350, :] = 128
    ridges_frames.append(ridges)

    plt.imshow(ridges)
    plt.show()



# show images side by side, in single row
cmap = plt.cm.gray
plt.rcParams["axes.titlesize"] = "small"
axes = plt.figure(figsize=(8,8)).subplots(1, 6)
for i, ridges_treshold in enumerate(ridges_frames):
    axes[i].imshow(ridges_treshold)
    axes[i].set_title(f"Ridges {i+9}")
    axes[i].set_xticks([])
    axes[i].set_yticks([])
plt.show()




exit(0)




cmap = plt.cm.gray

plt.rcParams["axes.titlesize"] = "medium"
axes = plt.figure(figsize=(8,8)).subplots(2, 2)

meijering_image = meijering(image, black_ridges=True, sigmas=range(1, 7))
_, meijering_treshold = cv2.threshold(img_as_ubyte(meijering_image), 0, 255, cv2.THRESH_OTSU)
meijering_treshold = cv2.morphologyEx(meijering_treshold, cv2.MORPH_OPEN, np.ones((5, 5), dtype=np.uint8))

axes[0, 0].imshow(meijering_treshold, cmap=cmap)
axes[0, 0].set_title("Meijering")

sato_image = sato(image, black_ridges=True, sigmas=range(1, 7))
_, sato_treshold = cv2.threshold(img_as_ubyte(sato_image), 0, 255, cv2.THRESH_OTSU)
sato_treshold = cv2.morphologyEx(sato_treshold, cv2.MORPH_OPEN, np.ones((5, 5), dtype=np.uint8))
axes[0, 1].imshow(sato_treshold, cmap=cmap)
axes[0, 1].set_title("Sato")

frangi_image = frangi(image, black_ridges=True, sigmas=range(1, 7))
axes[1, 0].imshow(frangi_image, cmap=cmap)
axes[1, 0].set_title("Frangi")


axes[1, 1].imshow(original(image), cmap=cmap)
axes[1, 1].set_title("Original")

for ax in axes.ravel():
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()
