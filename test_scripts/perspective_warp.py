import os
from collections import deque
from threading import Thread
import cv2, time
import numpy as np
from scipy.signal import find_peaks

from utils import Utils

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
    src = 'rtsp://admin:Manzes1997@192.168.1.64:554/ISAPI/Streaming/Channels/101'
    src = 'rtsp://admin:Manzes1997@bicodigital.a.pinggy.link:18627/ISAPI/Streaming/Channels/101'
    #src = 'rtsp://admin:Manzes1997,@192.168.1.64:554/media/video2'
    #src = '../recordings/small_start.mp4'
    #src = '../recordings/small_normal.mp4'
    src = '../recordings/small_end.mp4'
    camera = cv2.VideoCapture(src)

    #named window
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    #cv2.setWindowProperty('frame', cv2.WND_PROP_TOPMOST, 1)


    fix_point_deque = deque(maxlen=400)

    # open all images in ../captures folder

    src_dir = '../captures/20240522/'
    image_files = [f for f in os.listdir(src_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()


    while True:
        try:
            _, frame = camera.read()
            #frame = cv2.imread(src_dir + image_files.pop(0))
            #frame = cv2.resize(frame, (1920, 1080))


            frame = Utils.warp_small_conveyer(frame)
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l_ch = lab[:, :, 0]
            gray = l_ch



            # when clicked on window, print coordinates
            def click_event(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    print(x, y)
            cv2.setMouseCallback('frame', click_event)



            point = (217, 347)
            point = (25, 728)
            point = (25, 512)
            #point = (1230, 869)

            radius = 12
            cv2.circle(frame, point, radius, (0, 0, 255), -1)
            cv2.circle(frame, point, 30, (0, 0, 255), 2)

            val_at_point = gray[point[1], point[0]]
            val_around_point = gray[point[1]-radius:point[1]+radius, point[0]-radius:point[0]+radius]
            val_around_point_mean = val_around_point.mean()


            fix_point_deque.append(255 - val_around_point_mean)





            #plot fix_point_deque with mean, min, max using matplotlib
            import matplotlib.pyplot as plt
            plt.clf()

            peaks, _ = find_peaks(np.array(fix_point_deque), distance=100)

            mean_peaks = np.array(fix_point_deque)[peaks].mean()
            min_inv = np.array(fix_point_deque).min()
            treshold = mean_peaks - (mean_peaks - min_inv) * 0.1

            plt.plot(np.array(fix_point_deque))
            plt.plot(peaks, np.array(fix_point_deque)[peaks], "x")
            plt.axhline(y=treshold, color='g', linestyle='--', label='Threshold')

            # auto show plot and resume
            plt.draw()
            plt.pause(0.001)
            #plt.show()


            if 255 - val_around_point_mean < treshold:
                cv2.circle(frame, point, 30, (0, 255, 0), 2)
                cv2.rectangle(frame, (point[0]-radius, point[1]-radius), (frame.shape[1], point[1]+radius), (0, 255, 0), 2)
            else:
                cv2.circle(frame, point, 30, (0, 0, 255), 2)


            cv2.imshow('frame', frame)





            key = cv2.waitKey(10)
            # of q
            if key == ord('q'):
                break
            if key == ord('s'):
                plt.show()
        except AttributeError:
            pass

#ffmpeg -rtsp_transport tcp -i "rtsp://admin:Manzes1997@192.168.1.64:554/ISAPI/Streaming/Channels/103" -c copy /home/matejnevlud/app/output.mp4