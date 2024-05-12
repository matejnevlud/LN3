import cv2
import numpy as np
from skimage import img_as_ubyte
from skimage.filters.ridges import sato
import time

CALC_TIME = True

W = 1280
H = 720

STRIP_X_POSITION = 600
HORIZONTAL_STRIP_Y_POSITION = 310

DETECTION_RADIUS = 15
DETECTION_COOLDOWN = 1.5


def warp_conveyer(frame):
    t = cv2.getTickCount()

    pts1_1280x720_screenshot = np.float32([[161, 21], [1116, 29], [81, 584], [1224, 589]])
    pts1_1280x720 = np.float32([[166, 163], [1103, 171], [66, 641], [1230, 645]])

    # edit points for any frame.shape size
    pts1 = np.float32(pts1_1280x720)
    pts1[:, 0] = pts1[:, 0] * frame.shape[1] / W
    pts1[:, 1] = pts1[:, 1] * frame.shape[0] / H

    pts1 = np.float32(pts1)
    pts2 = np.float32([[0, 0], [W, 0], [0, H], [W, H]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    frame = cv2.warpPerspective(frame, matrix, (W, H))
    if CALC_TIME:
        print(f"Warp: {int((cv2.getTickCount() - t) / cv2.getTickFrequency() * 1000)} ms")
    return frame


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
    if CALC_TIME:
        print(f"WB: {int((cv2.getTickCount() - t) / cv2.getTickFrequency() * 1000)} ms")
    return (frame * 255).astype(np.uint8)


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


def take_horizontal_strip(frame):
    return frame[HORIZONTAL_STRIP_Y_POSITION:HORIZONTAL_STRIP_Y_POSITION + 100, 0:W]


def take_strip(frame):
    return frame[0:H, STRIP_X_POSITION:STRIP_X_POSITION + 100]





def horizontal_gauss(frame):
    t = cv2.getTickCount()
    frame = cv2.GaussianBlur(frame, (61, 9), 51)
    print(f"Horizontal Gauss: {int((cv2.getTickCount() - t) / cv2.getTickFrequency() * 1000)} ms")
    return frame


def detect_ridges(frame):
    t = cv2.getTickCount()
    ridges = sato(frame, black_ridges=True, sigmas=[9, 10, 11])
    _, ridges_treshold = cv2.threshold(img_as_ubyte(ridges), 0, 255, cv2.THRESH_OTSU)
    ridges_treshold = cv2.morphologyEx(ridges_treshold, cv2.MORPH_OPEN, np.ones((7, 31), dtype=np.uint8))
    if CALC_TIME:
        print(f"Ridges: {int((cv2.getTickCount() - t) / cv2.getTickFrequency() * 1000)} ms")
    return ridges_treshold


class DetectionPoint:

    #def static method
    @staticmethod
    def init_points():
        detection_points = []
        for i in range(0, 18):
            x = int(W / 18 * i + (W / 18 / 2))
            y = 50
            detection_points.append(DetectionPoint(x, y))
        return detection_points

    def __init__(self, x, y):
        self.counter = 0
        self.x = x
        self.y = y

        self.is_detected = False
        self.detected_time = 0
        self.mask = None

    def update(self, mask):
        self.mask = mask
        # test if circle with radius 10 is in beige mask with center in point
        circle_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.circle(circle_mask, (self.x, self.y), DETECTION_RADIUS, 255, -1)
        subtracted = cv2.subtract(circle_mask, mask)

        # check if is detected
        if self.is_detected:
            if time.time() - self.detected_time > DETECTION_COOLDOWN:
                self.is_detected = False
            return

        # check if is detected
        if cv2.countNonZero(subtracted) == 0:
            self.is_detected = True
            self.detected_time = time.time()
            self.counter += 1
        else:
            self.is_detected = False

    def get_count(self):
        return self.counter

    def draw(self, frame):

        if self.is_detected:
            cv2.circle(frame, (self.x, self.y), DETECTION_RADIUS, (0, 255, 0), -1)
        else:
            cv2.circle(frame, (self.x, self.y), DETECTION_RADIUS, (0, 0, 255), -1)

        # draw count under the point
        cv2.putText(frame, str(self.counter), (self.x - 15, self.y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def __str__(self):
        return f"({self.x}, {self.y})"

    def __repr__(self):
        return self.__str__()
