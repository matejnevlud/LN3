import os

import cv2
import skimage
from skimage import io, img_as_ubyte
from skimage.exposure import match_histograms, equalize_hist
import matplotlib.pyplot as plt
# import the necessary packages
import numpy as np
import cv2


from utils import apply_reference_color, warp_conveyer_calculate


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


def threshold_noodles(frame_bgr):
    low_beige = (0, 20, 20)
    high_beige = (80, 180, 255)

    mask = cv2.inRange(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV), low_beige, high_beige)
    kernel = np.ones((9, 9), np.uint8)
    #mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.erode(mask, kernel, iterations=3)
    mask = cv2.dilate(mask, kernel, iterations=3)
    #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

def contour_noodles(mask, debug_frame=None):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(mask.shape, dtype=np.uint8)
    for contour in contours:
        hull = cv2.convexHull(contour)
        cv2.drawContours(mask, [hull], 0, 255, -1)
        if debug_frame is not None:
            cv2.drawContours(debug_frame, [hull], 0, (0, 255, 255), 2)

    return mask, debug_frame



# load all images in the folder, show them one by one using plot
src_dir = '/Users/matejnevlud/github/LN3/captures/14_05'
# Get list of image files in the directory
image_files = [f for f in os.listdir(src_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
image_files.sort()

reference = cv2.imread('/Users/matejnevlud/github/LN3/captures/14_05/20240514_144830.jpg')
# Loop through each image and display it
for image_file in image_files:
    # Open the image file
    img_path = os.path.join(src_dir, image_file)
    img = cv2.imread(img_path)

    #

    cv2.imshow('Origo', img)
    #frame = color_transfer(reference, img)
    region = warp_conveyer_calculate(img)
    region = cv2.GaussianBlur(region, (11, 11), 0)
    noodles_mask = threshold_noodles(region)
    #? detect contours and draw convex hull
    noodles_mask, region = contour_noodles(noodles_mask, region)

    cv2.imshow('Transfer', region)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

    continue


    hsv_image_h = hsv_image[:, :, 0]
    reference_h = cv2.cvtColor(reference, cv2.COLOR_BGR2HSV)[:, :, 0]
    matched_h = match_histograms(hsv_image_h, reference_h)


    #join channels
    matched_hsv = hsv_image
    matched_hsv[:, :, 0] = matched_h
    matched = cv2.cvtColor(matched_hsv, cv2.COLOR_HSV2BGR)


    def gray_world(image):
        """
        Returns a plot comparison of original and corrected/white balanced image
        using the Gray World algorithm.

        Parameters
        ----------
        image : numpy array
                Image to process using gray world algorithm
        """
        # Apply the Gray World algorithm
        image_grayworld = ((image * (image.mean() / image.mean(axis=(0, 1)))).clip(0, 255).astype(np.uint8))
        return image_grayworld
        # Exclude alpha or opacity channel (transparency)
        if image.shape[2] == 4:
            image_grayworld[:, :, 3] = 255



        # Plot the comparison between the original and gray world corrected images
        fig, ax = plt.subplots(1, 2, figsize=(14, 10))
        ax[0].imshow(image)
        ax[0].set_title('Original Image')
        ax[0].axis('off')

        ax[1].imshow(image_grayworld)
        ax[1].set_title('Gray World Corrected Image')
        ax[1].axis('off')

        plt.show()

    def white_patch(image, percentile=100):
        """
        Returns a plot comparison of original and corrected/white balanced image
        using the White Patch algorithm.

        Parameters
        ----------
        image : numpy array
                Image to process using white patch algorithm
        percentile : integer, optional
                      Percentile value to consider as channel maximum
        """
        white_patch_image = img_as_ubyte(
            (image * 1.0 / np.percentile(image,
                                         percentile,
                                         axis=(0, 1))).clip(0, 1))
        return white_patch_image

    # Call the function to apply the Gray World algorithm
    matched = gray_world(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    matched = white_patch(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 90)
    matched = cv2.cvtColor(matched, cv2.COLOR_RGB2BGR)

    matched = whitebalance(img)





    # Display the image using opencv, show first third from input image, second third from reference image and third third from output image
    #stitched = cv2.hconcat([img[:, :img.shape[1]//3], reference[:reference.shape[0]//3, :], matched[:matched.shape[0]//3, :]])
    stitched_vertical = cv2.vconcat([img[:img.shape[0]//3, :], reference[reference.shape[0]//3:2*reference.shape[0]//3, :], matched[2*matched.shape[0]//3:, :]])
    cv2.imshow('Stitched Vertical', matched)


    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

exit(0)

# Load the source and reference images
source = img_as_ubyte(io.imread('/Users/matejnevlud/github/LN3/captures/out_10_00.jpg'))
reference = img_as_ubyte(io.imread('/Users/matejnevlud/github/LN3/captures/out_09_51.jpg'))

# Match the histograms
#matched = match_histograms(source, reference, channel_axis=2)
matched = equalize_hist(source)

# Display the images
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),
                                    sharex=True, sharey=True)
for aa in (ax1, ax2, ax3):
    aa.set_axis_off()

ax1.imshow(source)
ax1.set_title('Source')
ax2.imshow(reference)
ax2.set_title('Reference')
ax3.imshow(matched)
ax3.set_title('Matched')

plt.tight_layout()
plt.show()