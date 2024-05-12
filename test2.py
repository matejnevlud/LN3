from threading import Thread
import cv2, time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
from skimage.morphology import skeletonize

from utils import warp_conveyer, take_strip, horizontal_gauss, detect_ridges, take_horizontal_strip, apply_reference_color, DetectionPoint


class ThreadedCamera(object):
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.FPS = 1/100
        self.FPS_MS = int(self.FPS * 1000)
        # First initialisation self.status and self.frame
        (self.status, self.frame) = self.capture.read()

        # Start frame retrieval thread
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
            time.sleep(self.FPS)


if __name__ == '__main__':
    # open named window and set it to be at the WND_PROP_TOPMOST
    cv2.namedWindow('noodles_mask', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('noodles_mask', cv2.WND_PROP_TOPMOST, 1)
    cv2.moveWindow('noodles_mask', 0, 0)

    cv2.namedWindow('gap_mask', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('gap_mask', cv2.WND_PROP_TOPMOST, 1)
    cv2.moveWindow('gap_mask', 0, 480)

    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('frame', cv2.WND_PROP_TOPMOST, 1)
    cv2.moveWindow('frame', 0, 150)

    cv2.namedWindow('frame_mask', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('frame_mask', cv2.WND_PROP_TOPMOST, 1)
    cv2.moveWindow('frame_mask', 0, 300)






    # show 3 sliders for hue, saturation and value lower and upper bounds
    def nothing(x):
        pass
    cv2.namedWindow('Trackbars')
    cv2.createTrackbar('HH', 'Trackbars', 0, 179, nothing)
    cv2.setTrackbarPos('HH', 'Trackbars', 70)
    cv2.createTrackbar('HL', 'Trackbars', 0, 179, nothing)
    cv2.setTrackbarPos('HL', 'Trackbars', 0)
    cv2.createTrackbar('SH', 'Trackbars', 0, 255, nothing)
    cv2.setTrackbarPos('SH', 'Trackbars', 255)
    cv2.createTrackbar('SL', 'Trackbars', 0, 255, nothing)
    cv2.createTrackbar('VH', 'Trackbars', 0, 255, nothing)
    cv2.setTrackbarPos('VH', 'Trackbars', 255)
    cv2.createTrackbar('VL', 'Trackbars', 0, 255, nothing)


    # detection points
    DETECTION_POINTS = DetectionPoint.init_points()


    def apply_blur(frame_bgr):
        frame_bgr = cv2.GaussianBlur(frame_bgr, (11, 11), 0)
        #frame_bgr = cv2.medianBlur(frame_bgr, 11)
        #frame_bgr = horizontal_gauss(frame_bgr)
        #frame_bgr = cv2.bilateralFilter(frame_bgr, 17, 128, 50)
        return frame_bgr

    def threshold_noodles(frame_bgr):
        low_beige = (0, 0, 50)
        high_beige = (76, 255, 235)
        if False :
            low_h = cv2.getTrackbarPos('HL', 'Trackbars')
            low_s = cv2.getTrackbarPos('SL', 'Trackbars')
            low_v = cv2.getTrackbarPos('VL', 'Trackbars')
            high_h = cv2.getTrackbarPos('HH', 'Trackbars')
            high_s = cv2.getTrackbarPos('SH', 'Trackbars')
            high_v = cv2.getTrackbarPos('VH', 'Trackbars')
            low_beige = (low_h, low_s, low_v)
            high_beige = (high_h, high_s, high_v)

        mask = cv2.inRange(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV), low_beige, high_beige)
        kernel = np.ones((11, 11), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    def contour_noodles(mask, debug_frame=None):
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros(mask.shape, dtype=np.uint8)
        for contour in contours:
            hull = cv2.convexHull(contour)
            cv2.drawContours(mask, [hull], 0, 255, -1)
            if debug_frame is not None:
                cv2.drawContours(debug_frame, [hull], 0, (0, 255, 0), 2)

        return mask, debug_frame

    src = '/Users/matejnevlud/github/LN3/recordings/out_09_53.mp4'
    capture = cv2.VideoCapture(src)
    while True:
        try:
            #!frame is BGR !!!
            ret, frame = capture.read()

            #? apply reference color
            #frame = apply_reference_color(frame)

            #? warp frame to get rid of perspective
            frame = warp_conveyer(frame)

            #? take horizontal strip of frame
            region = take_horizontal_strip(frame)

            #? apply blurring for better color detection
            region = apply_blur(region)

            #? create mask for noodles, close and open it to remove noise
            noodles_mask = threshold_noodles(region)

            #? detect contours and draw convex hull
            noodles_mask, region = contour_noodles(noodles_mask, region)


            #? update state for detection points
            for point in DETECTION_POINTS:
                point.update(noodles_mask)
                point.draw(region)



            cv2.imshow('frame', region)


            # wait key 30 ms, if q, exit
            if cv2.waitKey(30) & 0xFF == ord('q'):
                exit(0)

        except AttributeError:
            pass
            exit(-1)

exit(0)



src = 'rtsp://admin:Manzes1997@192.168.1.64:554/ISAPI/Streaming/Channels/102'
threaded_camera = ThreadedCamera(src)
while True:
    try:
        cv2.imshow('frame', threaded_camera.frame)
        cv2.waitKey(threaded_camera.FPS_MS)
    except AttributeError:
        pass
