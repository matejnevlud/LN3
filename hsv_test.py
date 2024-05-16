import os
from threading import Thread
import cv2, time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors

src_dir = '/Users/matejnevlud/github/LN3/hsv_test/'

frame9 = cv2.imread(src_dir + '13.jpeg')

frame9 = cv2.imread('/Users/matejnevlud/github/LN3/captures/20240515 /20240515_123820.jpg')
img = cv2.imread('/Users/matejnevlud/github/LN3/captures/20240515 /20240515_124826.jpg')


src_dir = '/Users/matejnevlud/github/LN3/captures/20240515 /'
image_files = [f for f in os.listdir(src_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

for image_file in image_files:
    img_path = os.path.join(src_dir, image_file)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (640, 480))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    pixel_colors = img.reshape((np.shape(img)[0]*np.shape(img)[1], 3))
    norm = colors.Normalize(vmin=-1.,vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()


    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    mask_out = cv2.inRange(hsv_img, (60, 0, 0), (150, 255, 255))
    inverted_mask = cv2.bitwise_not(mask_out)
    img = cv2.bitwise_and(img, img, mask=inverted_mask)

    hsv_img = cv2.bitwise_and(hsv_img, hsv_img, mask=inverted_mask)

    h, s, v = cv2.split(hsv_img)

    #set fullscreen plot
    plt.figure(figsize=(16, 10))
    axis = plt.axes(projection="3d")


    axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Hue")
    axis.set_ylabel("Saturation")
    axis.set_zlabel("Value")
    # flip camera view elevation 23, azimuth 81, roll 0
    axis.view_init(elev=23, azim=120)

    # behind the scatter points show a 2D histogram
    plt.figure(figsize=(16, 10))
    plt.imshow(img)


    # show fullscreen plot
    plt.show()
