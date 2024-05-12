from threading import Thread
import cv2, time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors

src_dir = '/Users/matejnevlud/github/LN3/hsv_test/'

frame9 = cv2.imread(src_dir + '13.jpeg')

frame9 = cv2.imread('/Users/matejnevlud/github/LN3/captures/out_09_45.jpg')
frame9 = cv2.imread('/Users/matejnevlud/github/LN3/dual.png')

width = frame9.shape[1]
height = frame9.shape[0]


frame9 = cv2.cvtColor(frame9, cv2.COLOR_BGR2RGB)
pixel_colors = frame9.reshape((np.prod(frame9.shape[:2]), 3))
plt.imshow(frame9)
plt.show()
frame9 = cv2.cvtColor(frame9, cv2.COLOR_RGB2HSV)
plt.imshow(frame9)
plt.show()

# split the frame into three channels, normalize the hue channel
h = frame9[:,:,0] # hue but 0 - 179
s = frame9[:,:,1]
v = frame9[:,:,2]

# plot all three channels as separate 2D images, under each one show hisogram of channel values
fig, ax = plt.subplots(1, 6, figsize=(15, 5))
ax[0].imshow(h, cmap='hsv')
ax[0].set_title('Hue')
ax[1].imshow(s, cmap='hsv')
ax[1].set_title('Saturation')
ax[2].imshow(v, cmap='hsv')
ax[2].set_title('Value')

ax[3].hist(h.ravel(), bins=180, color='r', alpha=0.5)
ax[3].set_title('Hue histogram')
ax[4].hist(s.ravel(), bins=256, color='g', alpha=0.5)
ax[4].set_title('Saturation histogram')
ax[5].hist(v.ravel(), bins=256, color='b', alpha=0.5)
ax[5].set_title('Value histogram')

plt.show()


# plot Saturation as a function of xy grid in 3D, draw original image as a surface
X, Y = np.meshgrid(np.arange(s.shape[1]), np.arange(s.shape[0]))
Z = h
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='hsv')
plt.show()