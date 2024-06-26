import os

import cv2
import skimage
from skimage import io, img_as_ubyte
from skimage.exposure import match_histograms, equalize_hist
import matplotlib.pyplot as plt
# import the necessary packages
import numpy as np
import cv2

from utils import whitebalance


def image_stats(image):
    # compute the mean and standard deviation of each channel
    (l, a, b) = cv2.split(image)
    (lMean, lStd) = (l.mean(), l.std())
    (aMean, aStd) = (a.mean(), a.std())
    (bMean, bStd) = (b.mean(), b.std())
    # return the color statistics
    return (lMean, lStd, aMean, aStd, bMean, bStd)

def color_transfer(source, target):
    # convert the images from the RGB to L*ab* color space, being
    # sure to utilizing the floating point data type (note: OpenCV
    # expects floats to be 32-bit, so use that instead of 64-bit)
    #source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")
    # compute color statistics for the source and target images
    (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = image_stats(source)
    (lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = image_stats(target)
    # subtract the means from the target image
    (l, a, b) = cv2.split(target)
    l -= lMeanTar
    a -= aMeanTar
    b -= bMeanTar
    # scale by the standard deviations
    l = (lStdTar / lStdSrc) * l
    a = (aStdTar / aStdSrc) * a
    b = (bStdTar / bStdSrc) * b
    # add in the source mean
    l += lMeanSrc
    a += aMeanSrc
    b += bMeanSrc
    # clip the pixel intensities to [0, 255] if they fall outside
    # this range
    l = np.clip(l, 0, 255)
    a = np.clip(a, 0, 255)
    b = np.clip(b, 0, 255)
    # merge the channels together and convert back to the RGB color
    # space, being sure to utilize the 8-bit unsigned integer data
    # type
    transfer = cv2.merge([l, a, b])
    transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)

    # return the color transferred image
    return transfer


def apply_reference_color(target):
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Precompute means and standard deviations
    lMeanSrc, lStdSrc = 110.5742, 51.16022
    aMeanSrc, aStdSrc = 126.81388, 2.2345092
    bMeanSrc, bStdSrc = 125.23372, 7.068241

    l, a, b = cv2.split(target_lab)

    lMeanTar, lStdTar = l.mean(), l.std()
    aMeanTar, aStdTar = a.mean(), a.std()
    bMeanTar, bStdTar = b.mean(), b.std()

    # Perform color transfer directly on LAB channels
    l = ((l - lMeanTar) * (lStdSrc / lStdTar)) + lMeanSrc
    a = ((a - aMeanTar) * (aStdSrc / aStdTar)) + aMeanSrc
    b = ((b - bMeanTar) * (bStdSrc / bStdTar)) + bMeanSrc

    # Clip pixel intensities
    l = np.clip(l, 0, 255)
    a = np.clip(a, 0, 255)
    b = np.clip(b, 0, 255)

    transfer_lab = cv2.merge([l, a, b])
    transfer_bgr = cv2.cvtColor(transfer_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    return transfer_bgr




# load all images in the folder, show them one by one using plot
src_dir = '/Users/matejnevlud/Downloads/frame_20240510_122409.jpg/'
src_dir = '/captures/14_05/'
# Get list of image files in the directory
image_files = [f for f in os.listdir(src_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
image_files.sort()

reference = cv2.imread('/Users/matejnevlud/github/LN3/captures/14_05/20240514_144830.jpg')
# Loop through each image and display it
for image_file in image_files:
    # Open the image file
    img_path = os.path.join(src_dir, image_file)
    img = cv2.imread(img_path)



    #img = color_transfer(reference, img)
    img = color_transfer(reference, img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # only yellow part
    yellow = b > 128
    yellow = yellow.astype(np.uint8) * 255

    yellow = cv2.dilate(yellow, np.ones((5, 5), np.uint8), iterations=1)
    yellow = cv2.erode(yellow, np.ones((5, 5), np.uint8), iterations=4)

    # close the gaps
    yellow = cv2.morphologyEx(yellow, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))





    # set plot size
    plt.figure(figsize=(16, 9))
    plt.imshow(yellow)
    plt.show()

exit(0)
