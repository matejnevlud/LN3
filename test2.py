from threading import Thread
import cv2, time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
from skimage.morphology import skeletonize
from collections import deque
from utils import HORIZONTAL_STRIP_Y_POSITION, warp_conveyer, warp_conveyer_calculate, horizontal_gauss, detect_ridges, take_horizontal_strip, apply_reference_color, DetectionPoint, color_transfer, whitebalance


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



frametime_deque = deque(maxlen=30)

if __name__ == '__main__':
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('frame', cv2.WND_PROP_TOPMOST, 1)
    #cv2.moveWindow('frame', 0, 150)

    # detection points
    DETECTION_POINTS = DetectionPoint.init_points()


    def apply_blur(frame_bgr):
        frame_bgr = cv2.GaussianBlur(frame_bgr, (11, 11), 0)
        #frame_bgr = cv2.medianBlur(frame_bgr, 11)
        #frame_bgr = horizontal_gauss(frame_bgr)
        #frame_bgr = cv2.bilateralFilter(frame_bgr, 17, 128, 50)
        return frame_bgr

    def threshold_noodles(frame_bgr):
        low_beige = (0, 0, 20)
        high_beige = (76, 255, 255)


        mask_inverted = cv2.inRange(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV), (65, 0, 0), (150, 255, 255))
        mask = cv2.bitwise_not(mask_inverted)
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


    reference = cv2.imread('/Users/matejnevlud/github/LN3/captures/14_05/20240514_144830.jpg')
    src = '/Users/matejnevlud/github/LN3/recordings/out_09_53.mp4'
    #src = 'rtsp://admin:Manzes1997@192.168.1.64:554/ISAPI/Streaming/Channels/101'
    src = 'rtsp://admin:Manzes1997@7.tcp.eu.ngrok.io:12706/ISAPI/Streaming/Channels/101'
    src = 'rtsp://admin:Manzes1997@bicodigital.a.pinggy.link:18627/ISAPI/Streaming/Channels/101'
    #capture = cv2.VideoCapture(src)
    threaded_camera = ThreadedCamera(src)
    #capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    while True:
        try:
            t = cv2.getTickCount()
            #!frame is BGR !!!
            frame = threaded_camera.frame

            #? apply reference color
            #frame = whitebalance(frame)
            #frame = apply_reference_color(frame)
            #frame = color_transfer(reference, frame)

            #? warp frame to get rid of perspective
            frame = warp_conveyer_calculate(frame)

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



            #TODO : ONLY DEBUG
            whole_frame_noodles_mask, frame = contour_noodles(threshold_noodles(apply_blur(frame)), frame)


            # paste region onto frame at position 0, 0
            frame[HORIZONTAL_STRIP_Y_POSITION:region.shape[0] + HORIZONTAL_STRIP_Y_POSITION, 0:region.shape[1]] = region
            cv2.rectangle(frame, (0, HORIZONTAL_STRIP_Y_POSITION), (frame.shape[1], region.shape[0]+HORIZONTAL_STRIP_Y_POSITION), (0, 0, 255), 3)





            frametime_deque.append(int((cv2.getTickCount() - t) / cv2.getTickFrequency() * 1000))
            cv2.putText(frame, f"FPS: {int(1000 / np.mean(frametime_deque))}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('frame', frame)






            key = cv2.waitKey(threaded_camera.FPS_MS) & 0xFF

            #if h is pressed, show histogram of the frame in matplotlib
            if key == ord('h'):
                # histogram of HSV only hue

                small_frame = cv2.resize(frame, (640, 480))
                hsv = cv2.cvtColor(small_frame, cv2.COLOR_RGB2HSV)
                h, s, v = cv2.split(hsv)

                plt.figure(figsize=(16, 10))
                axis = plt.axes(projection="3d")

                pixel_colors = small_frame.reshape((np.shape(small_frame)[0]*np.shape(small_frame)[1], 3))
                norm = colors.Normalize(vmin=-1.,vmax=1.)
                norm.autoscale(pixel_colors)
                pixel_colors = norm(pixel_colors).tolist()


                axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
                axis.set_xlabel("Hue")
                axis.set_ylabel("Saturation")
                axis.set_zlabel("Value")
                # flip camera view elevation 23, azimuth 81, roll 0
                axis.view_init(elev=23, azim=120)

                plt.show()

            # wait key 30 ms, if q, exit
            if key == ord('q'):
                exit(0)


        except AttributeError:
            pass
            exit(-1)

exit(0)



