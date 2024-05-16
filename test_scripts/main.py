from threading import Thread
import cv2, time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from utils import warp_conveyer, take_strip, horizontal_gauss, detect_ridges, apply_reference_color

src = '/recordings/out_10_03.mp4'

capture = cv2.VideoCapture(src)
capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

while True:
    (status, frame) = capture.read()
    if not status:
        break

    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = apply_reference_color(frame)
    frame = warp_conveyer(frame)

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)


    frame_gray = take_strip(frame_gray)
    frame_gray = horizontal_gauss(frame_gray)
    frame_gray = detect_ridges(frame_gray)


    # show frame
    cv2.imshow('frame', frame)
    cv2.imshow('frame_gray', frame_gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break