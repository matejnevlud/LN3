import sqlite3
from datetime import date
from threading import Thread
import cv2, time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
from skimage.morphology import skeletonize
from collections import deque
#from utils import HORIZONTAL_STRIP_Y_POSITION, warp_conveyer, warp_conveyer_calculate, horizontal_gauss, detect_ridges, take_horizontal_strip, apply_reference_color, DetectionPoint, color_transfer, whitebalance
import cv2
import numpy as np
from skimage import img_as_ubyte
from skimage.filters.ridges import sato
import time


IS_DEBUG = False

W = 1920
H = 1080

STRIP_X_POSITION = 600
HORIZONTAL_STRIP_Y_POSITION = 310

DETECTION_RADIUS = 27
DETECTION_COOLDOWN = 1.3

MOVING_COUNT_THRESHOLD = 200
MOVING_MEAN_THRESHOLD = 0.8


class Utils:

    @staticmethod
    def warp_conveyer_calculate(frame):
        t = cv2.getTickCount()

        pts1_1280x720_screenshot = np.float32([[161, 21], [1116, 29], [81, 584], [1224, 589]])
        pts1_1280x720 = np.float32([[166, 163], [1103, 171], [66, 641], [1230, 645]])
        pts1_1920x1080 = np.float32([[261, 204], [1642, 213], [102, 931], [1845, 942]])
        pts1_2688x1520 = np.float32([[326, 120], [2356, 147], [150, 1327], [2591, 1336]])
        # edit points for any frame.shape size
        if frame.shape[1] == 1280 and frame.shape[0] == 720:
            pts1 = pts1_1280x720
        elif frame.shape[1] == 1920 and frame.shape[0] == 1080:
            pts1 = pts1_1920x1080
        elif frame.shape[1] == 2688 and frame.shape[0] == 1520:
            pts1 = pts1_2688x1520
        else:
            pts1 = pts1_1280x720_screenshot


        pts1 = np.float32(pts1)
        pts2 = np.float32([[0, 0], [W, 0], [0, H], [W, H]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)

        frame = cv2.warpPerspective(frame, matrix, (W, H))
        if IS_DEBUG:
            print(f"Warp DYN: {int((cv2.getTickCount() - t) / cv2.getTickFrequency() * 1000)} ms")
        return frame

    @staticmethod
    def warp_conveyer(frame):
        t = cv2.getTickCount()
        matrix = np.array([[9.78465021e-01, 2.04699795e-01, -2.93686890e+02],
                           [-1.15744296e-02, 1.35565506e+00, -3.28575630e+02],
                           [-1.08687033e-05, 3.67648301e-04, 1.00000000e+00]])
        if frame.shape[1] == 1920 and frame.shape[0] == 1080:
            matrix = np.array([[ 1.49669178e+00,  3.27336992e-01, -4.57413301e+02],
                               [-1.31601690e-02,  2.01935481e+00, -4.08513578e+02],
                               [-2.83612875e-06,  3.88347700e-04,  1.00000000e+00]])

        # use,  INTER_NEAREST
        frame = cv2.warpPerspective(frame, matrix, (W, H))
        if IS_DEBUG:
            print(f"Warp STAT: {int((cv2.getTickCount() - t) / cv2.getTickFrequency() * 1000)} ms")
        return frame

    @staticmethod
    def whitebalance(frame):
        t = cv2.getTickCount()

        def gamma_decompress(im):
            return np.power(im, 2.2)

        def gamma_compress(im):
            return np.power(im, 1 / 2.2)

        def measure_gray_world(im):
            return np.mean(im, axis=(0, 1))

        frame = gamma_decompress(frame / 255)
        avg = measure_gray_world(frame)
        frame = frame / avg * 0.3
        frame = gamma_compress(frame)
        if IS_DEBUG:
            print(f"WB: {int((cv2.getTickCount() - t) / cv2.getTickFrequency() * 1000)} ms")
        return (frame * 255).astype(np.uint8)



    @staticmethod
    def image_stats(image):
        # compute the mean and standard deviation of each channel
        (l, a, b) = cv2.split(image)
        (lMean, lStd) = (l.mean(), l.std())
        (aMean, aStd) = (a.mean(), a.std())
        (bMean, bStd) = (b.mean(), b.std())
        # return the color statistics
        return (lMean, lStd, aMean, aStd, bMean, bStd)
    @staticmethod
    def color_transfer(source, target):
        # convert the images from the RGB to L*ab* color space, being
        # sure to utilizing the floating point data type (note: OpenCV
        # expects floats to be 32-bit, so use that instead of 64-bit)
        source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
        target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")
        # compute color statistics for the source and target images
        (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = Utils.image_stats(source)
        (lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = Utils.image_stats(target)
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
    @staticmethod
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

    @staticmethod
    def take_horizontal_strip(frame):
        return frame[HORIZONTAL_STRIP_Y_POSITION:HORIZONTAL_STRIP_Y_POSITION + 200, 0:W]

    @staticmethod
    def take_strip(frame):
        return frame[0:H, STRIP_X_POSITION:STRIP_X_POSITION + 100]

    @staticmethod
    def horizontal_gauss(frame):
        t = cv2.getTickCount()
        frame = cv2.GaussianBlur(frame, (61, 9), 51)
        print(f"Horizontal Gauss: {int((cv2.getTickCount() - t) / cv2.getTickFrequency() * 1000)} ms")
        return frame

    @staticmethod
    def detect_ridges(frame):
        t = cv2.getTickCount()
        ridges = sato(frame, black_ridges=True, sigmas=[9, 10, 11])
        _, ridges_treshold = cv2.threshold(img_as_ubyte(ridges), 0, 255, cv2.THRESH_OTSU)
        ridges_treshold = cv2.morphologyEx(ridges_treshold, cv2.MORPH_OPEN, np.ones((7, 31), dtype=np.uint8))
        if IS_DEBUG:
            print(f"Ridges: {int((cv2.getTickCount() - t) / cv2.getTickFrequency() * 1000)} ms")
        return ridges_treshold

class DetectionPoint:

    #def static method
    @staticmethod
    def init_points(database):
        detection_points = []
        for i in range(0, 18):
            x = int(W / 18 * i + (W / 18 / 2))
            y = 50
            detection_points.append(DetectionPoint(i, x, y, database))
        return detection_points

    def __init__(self, region_id, x, y, database):
        self.counter = 0
        self.x = x
        self.y = y
        self.database = database

        self.region_id = region_id
        self.is_detected = False
        self.detected_time = 0
        self.mask = None

    def update(self, mask):
        self.mask = mask
        # test if circle with radius 10 is in beige mask with center in point
        circle_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.circle(circle_mask, (self.x, self.y), DETECTION_RADIUS, 255, -1)
        subtracted = cv2.subtract(circle_mask, mask)

        # check if is already detected
        if self.is_detected:
            if time.time() - self.detected_time > DETECTION_COOLDOWN:
                self.is_detected = False
            return

        if not is_moving:
            return

        # check if is detected
        if cv2.countNonZero(subtracted) == 0:
            self.is_detected = True
            self.detected_time = time.time()
            self.counter += 1
            self.database.add_detection(self.region_id)
        else:
            self.is_detected = False

    def get_count(self):
        return self.counter

    def draw(self, frame):
        if not is_moving:
            cv2.circle(frame, (self.x, self.y), DETECTION_RADIUS, (0, 140, 255), -1)
        elif self.is_detected:
            cv2.circle(frame, (self.x, self.y), DETECTION_RADIUS, (0, 255, 0), -1)
        else:
            cv2.circle(frame, (self.x, self.y), DETECTION_RADIUS, (0, 0, 255), -1)

        # draw count under the point
        cv2.putText(frame, str(self.counter), (self.x - 15, self.y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def __str__(self):
        return f"({self.x}, {self.y})"

    def __repr__(self):
        return self.__str__()

class Database:
    def __init__(self):
        self.conn = sqlite3.connect('detections.db')
        self.cursor = self.conn.cursor()
        # create table with self incrementing primary key id,
        # region_id, date and count of detections on that date
        self.cursor.execute('CREATE TABLE IF NOT EXISTS detections (id INTEGER PRIMARY KEY AUTOINCREMENT, region_id INTEGER, date TEXT, count INTEGER)')

        #also create table for logging start and stop of movement, with timestamp
        self.cursor.execute('CREATE TABLE IF NOT EXISTS movement (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, is_moving INTEGER)')

        # create view, which will show sum of detections for each date
        self.cursor.execute('CREATE VIEW IF NOT EXISTS detections_by_date AS SELECT date, SUM(count) FROM detections GROUP BY date')

    def add_detection(self, region_id):
        # date in format YYYY-MM-DD
        current_date = date.today().isoformat()
        self.cursor.execute('SELECT count FROM detections WHERE region_id = ? AND date = ?', (region_id, current_date))
        row = self.cursor.fetchone()
        if row is None:
            self.cursor.execute('INSERT INTO detections (region_id, date, count) VALUES (?, ?, 1)', (region_id, current_date))
        else:
            self.cursor.execute('UPDATE detections SET count = count + 1 WHERE region_id = ? AND date = ?', (region_id, current_date))
        self.conn.commit()

    def get_detections(self):
        self.cursor.execute('SELECT * FROM detections')
        return self.cursor.fetchall()

    def get_detections_by_date(self, date_iso):
        self.cursor.execute('SELECT * FROM detections WHERE date = ?', (date_iso,))
        return self.cursor.fetchall()

    def add_movement(self, _is_moving):
        self.cursor.execute('INSERT INTO movement (timestamp, is_moving) VALUES (?, ?)', (time.time(), 1 if _is_moving else 0))
        self.conn.commit()


    def close(self):
        self.conn.close()

class ThreadedCamera(object):
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.FPS = 30
        self.SLEEP_SECONDS = 1 / (self.FPS * 2)
        self.FPS_MS = int(1000 / self.FPS)
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
            time.sleep(self.SLEEP_SECONDS)


frametime_deque = deque(maxlen=30)

if __name__ == '__main__':
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('frame', cv2.WND_PROP_TOPMOST, 1)



    def apply_blur(frame_bgr):
        frame_bgr = cv2.GaussianBlur(frame_bgr, (11, 11), 0)
        #frame_bgr = cv2.medianBlur(frame_bgr, 11)
        #frame_bgr = horizontal_gauss(frame_bgr)
        #frame_bgr = cv2.bilateralFilter(frame_bgr, 17, 128, 50)
        return frame_bgr

    def threshold_noodles(frame_bgr):
        low_beige = (0, 0, 20)
        high_beige = (76, 255, 255)


        mask_inverted = cv2.inRange(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV), (60, 0, 0), (150, 255, 255))
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


    mog2 = cv2.createBackgroundSubtractorMOG2()
    mog2.setHistory(100)
    mog2.setVarThreshold(16)
    mog2.setDetectShadows(False)
    white_pixels = 0
    was_moving_deque = deque(maxlen=100)
    is_moving = False

    def determine_movement(region):
        square = region[0:region.shape[0], int(region.shape[1]/2 - region.shape[0]/2):int(region.shape[1]/2 + region.shape[0]/2)]
        fgmask = mog2.apply(square)
        fgmask = cv2.resize(fgmask, (50, 50))
        if IS_DEBUG:
            cv2.imshow('MOG movement mask', fgmask)
        # count white pixels
        global white_pixels
        white_pixels = cv2.countNonZero(fgmask)
        was_moving_deque.append(white_pixels > MOVING_COUNT_THRESHOLD)

        # did change from moving to not moving or vice versa
        global is_moving
        new_is_moving = np.mean(was_moving_deque) > MOVING_MEAN_THRESHOLD and len(was_moving_deque) == was_moving_deque.maxlen
        if new_is_moving != is_moving:
            is_moving = new_is_moving
            db.add_movement(new_is_moving)




    reference = cv2.imread('/Users/matejnevlud/github/LN3/captures/14_05/20240514_144830.jpg')
    src = 'rtsp://admin:Manzes1997@bicodigital.a.pinggy.link:18627/ISAPI/Streaming/Channels/101'
    src = 'recordings/out_08_07.mp4'

    # determine if running on raspberry pi
    try:
        import RPi.GPIO as GPIO
        src = 'rtsp://admin:Manzes1997@192.168.1.64:554/ISAPI/Streaming/Channels/101'
    except ImportError:
        pass

    threaded_camera = ThreadedCamera(src)
    db = Database()
    detection_points = DetectionPoint.init_points(db)

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
            frame = Utils.warp_conveyer_calculate(frame)

            #? take horizontal strip of frame
            region = Utils.take_horizontal_strip(frame)

            #? apply blurring for better color detection
            region = apply_blur(region)

            #? take square of regions height from the center of the region
            determine_movement(region)



            #? create mask for noodles, close and open it to remove noise
            noodles_mask = threshold_noodles(region)

            #? detect contours and draw convex hull
            noodles_mask, region = contour_noodles(noodles_mask, region)



            #? update state for detection points
            for point in detection_points:
                point.update(noodles_mask)
                point.draw(region)



            #TODO : ONLY DEBUG
            if IS_DEBUG:
                whole_frame_noodles_mask, frame = contour_noodles(threshold_noodles(apply_blur(frame)), frame)


            # paste region onto frame at position 0, 0
            frame[HORIZONTAL_STRIP_Y_POSITION:region.shape[0] + HORIZONTAL_STRIP_Y_POSITION, 0:region.shape[1]] = region
            cv2.rectangle(frame, (0, HORIZONTAL_STRIP_Y_POSITION), (frame.shape[1], region.shape[0]+HORIZONTAL_STRIP_Y_POSITION), (0, 0, 255), 3)

            frametime_deque.append(int((cv2.getTickCount() - t) / cv2.getTickFrequency() * 1000))
            cv2.putText(frame, f"FPS: {int(1000 / np.mean(frametime_deque))}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"IS MOVING {white_pixels}" if is_moving else f"NOT MOVING {white_pixels}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if is_moving else (0, 0, 255), 2)

            print(f"FPS: {int(1000 / np.mean(frametime_deque))}    ", end='\r')
            cv2.imshow('frame', frame)






            key = cv2.waitKey(threaded_camera.FPS_MS // 2) & 0xFF

            if key == ord('d'):
                IS_DEBUG = not IS_DEBUG
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



