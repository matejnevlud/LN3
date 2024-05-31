import concurrent
from collections import deque

import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import img_as_ubyte
from skimage.filters.ridges import sato
from config import IS_DEBUG, W, H, STRIP_X_POSITION, HORIZONTAL_STRIP_Y_START, HORIZONTAL_STRIP_HEIGHT
from scipy.signal import find_peaks

SPACES_STRIP_POSITION_X = 865 - 100
SPACES_STRIP_WIDTH = 80
SPACES_STRIP_KERNEL_HEIGHT = 30
SPACES_DISTANCE = 100


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

        frame = cv2.warpPerspective(frame, matrix, (W, H), flags=cv2.INTER_NEAREST)
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
            matrix = np.array([[1.49669178e+00, 3.27336992e-01, -4.57413301e+02],
                               [-1.31601690e-02, 2.01935481e+00, -4.08513578e+02],
                               [-2.83612875e-06, 3.88347700e-04, 1.00000000e+00]])

        # use,  INTER_NEAREST
        frame = cv2.warpPerspective(frame, matrix, (W, H))
        if IS_DEBUG:
            print(f"Warp STAT: {int((cv2.getTickCount() - t) / cv2.getTickFrequency() * 1000)} ms")
        return frame

    @staticmethod
    def warp_b(frame):
        t = cv2.getTickCount()

        pts1 = np.float32([[990 - 50, 253], [1752, 272], [952 - 50, 953], [1761, 972]])
        pts2 = np.float32([[0, 0], [H, 0], [0, H], [H, H]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)

        frame = cv2.warpPerspective(frame, matrix, (H, H), flags=cv2.INTER_NEAREST)
        if IS_DEBUG:
            print(f"Warp DYN: {int((cv2.getTickCount() - t) / cv2.getTickFrequency() * 1000)} ms")
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
    def kelvin_to_rgb(kelvin):
        kelvin = kelvin / 100.0
        if kelvin <= 66:
            r = 255
            g = 99.4708025861 * np.log(kelvin) - 161.1195681661
            b = 138.5177312231 * np.log(kelvin - 10) - 305.0447927307 if kelvin > 19 else 0
        else:
            r = 329.698727446 * ((kelvin - 60) ** -0.1332047592)
            g = 288.1221695283 * ((kelvin - 60) ** -0.0755148492)
            b = 255
        return np.clip([r, g, b], 0, 255)

    @staticmethod
    def adjust_color_temperature(image, kelvin):
        # Convert the input image to float
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0

        if kelvin == 8000:
            rgb_scale = np.array([0.86750843, 0.90113407, 1.])
        else:
            # Create a color temperature matrix based on the given Kelvin value
            rgb_scale = Utils.kelvin_to_rgb(kelvin)
            rgb_scale /= np.max(rgb_scale)  # Normalize the RGB scale

        # Apply the scale to the image
        result = image * rgb_scale

        # Ensure the result is within the valid range [0, 1]
        result = np.clip(result, 0, 1)

        # Convert the result back to uint8
        result = (result * 255).astype(np.uint8)

        return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    @staticmethod
    def adjust_color_temperature_8000_fast(frame):
        #frame *= np.array([0.86750843, 0.90113407, 1.]) RGB
        scale_factors = np.array([1.0, 0.90113407, 0.8675084])

        # Split the frame into B, G, R channels
        b, g, r = cv2.split(frame)

        # Apply scaling factors
        b = cv2.multiply(b, scale_factors[0])
        g = cv2.multiply(g, scale_factors[1])
        r = cv2.multiply(r, scale_factors[2])

        # Merge the channels back together
        frame = cv2.merge([b, g, r])
        return frame




    @staticmethod
    def threshold_noodles(frame_bgr):
        low_beige = (0, 0, 20)
        high_beige = (76, 255, 255)


        mask_inverted = cv2.inRange(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV), (60, 0, 0), (150, 255, 255))
        mask = cv2.bitwise_not(mask_inverted)

        # remove dark spots
        mask_dark = cv2.inRange(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV), (0, 0, 0), (180, 255, 50))
        mask = cv2.subtract(mask, mask_dark)

        kernel = np.ones((9, 9), np.uint8)
        #mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.erode(mask, kernel, iterations=3)
        mask = cv2.dilate(mask, kernel, iterations=3)
        #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    @staticmethod
    def contour_noodles(mask, debug_frame=None):
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        new_mask = np.zeros(mask.shape, dtype=np.uint8)
        for contour in contours:
            hull = cv2.convexHull(contour)
            # estimate curve, should be rectangle
            cv2.drawContours(new_mask, [hull], 0, 255, -1)
            if debug_frame is not None:
                cv2.drawContours(debug_frame, [hull], 0, (0, 255, 255), 2)


        return mask, debug_frame




    @staticmethod
    def add_frame_to_queue(frame, q):

        frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2)) # resize to 50% of original
        jpeg_50_quality = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])[1].tobytes()
        if q and not q.full():
            q.put(jpeg_50_quality)

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
    def take_horizontal_strip(frame, y_start=HORIZONTAL_STRIP_Y_START, height=HORIZONTAL_STRIP_HEIGHT):
        return frame[y_start:y_start + height, 0:W]

    @staticmethod
    def take_strip(frame, width=150):
        return frame[0:H, STRIP_X_POSITION:STRIP_X_POSITION + width]

    @staticmethod
    def horizontal_gauss(frame):
        t = cv2.getTickCount()
        frame = cv2.GaussianBlur(frame, (61, 9), 51)
        print(f"Horizontal Gauss: {int((cv2.getTickCount() - t) / cv2.getTickFrequency() * 1000)} ms")
        return frame

    @staticmethod
    def detect_ridges(frame):
        t = cv2.getTickCount()

        orig_size = frame.shape

        if frame.shape[2] > 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
        ridges = sato(frame, black_ridges=False, sigmas=[9, 10, 11])
        ridges = img_as_ubyte(ridges)
        #ridges = ridges.astype(np.uint8)

        _, ridges_treshold = cv2.threshold(ridges, 0, 255, cv2.THRESH_OTSU)
        ridges_treshold = cv2.morphologyEx(ridges_treshold, cv2.MORPH_OPEN, np.ones((7, 31), dtype=np.uint8))
        if IS_DEBUG:
            print(f"Ridges: {int((cv2.getTickCount() - t) / cv2.getTickFrequency() * 1000)} ms")

        # expand to original size
        ridges_treshold = cv2.resize(ridges_treshold, (orig_size[1], orig_size[0]))

        # countour and export only y coords
        contours, _ = cv2.findContours(ridges_treshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        y_coords = []
        for contour in contours:
            #get hull and find center y
            hull = cv2.convexHull(contour)
            M = cv2.moments(hull)
            cY = int(M["m01"] / M["m00"])
            y_coords.append(cY)

        return ridges_treshold, y_coords

    B_CONVEYER_DETECTION_MEASUREMENTS = deque(maxlen=200)
    B_CONVEYER_DETECTION_PEAKS_DISTANCE = 100
    B_CONVEYER_DETECTION_POINT = (25, 512)
    B_CONVEYER_DETECTION_WIDTH = 10
    B_CONVEYER_DETECTION_VALLEYS_DISTANCE = 100
    B_CONVEYER_DETECTION_THRESHOLD = 0.1

    @staticmethod
    def is_conveyer_in_target_b(frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_ch = lab[:, :, 0]

        values_around_point = l_ch[Utils.B_CONVEYER_DETECTION_POINT[1] - Utils.B_CONVEYER_DETECTION_WIDTH:Utils.B_CONVEYER_DETECTION_POINT[1] + Utils.B_CONVEYER_DETECTION_WIDTH,
                              Utils.B_CONVEYER_DETECTION_POINT[0] - Utils.B_CONVEYER_DETECTION_WIDTH:Utils.B_CONVEYER_DETECTION_POINT[0] + Utils.B_CONVEYER_DETECTION_WIDTH]
        values_around_point_mean = values_around_point.mean()
        Utils.B_CONVEYER_DETECTION_MEASUREMENTS.append(255 - values_around_point_mean)

        # find valleys
        valleys, _ = find_peaks(np.array(Utils.B_CONVEYER_DETECTION_MEASUREMENTS), distance=Utils.B_CONVEYER_DETECTION_PEAKS_DISTANCE)

        mean_valleys = np.array(Utils.B_CONVEYER_DETECTION_MEASUREMENTS)[valleys].mean()
        min_inv = np.array(Utils.B_CONVEYER_DETECTION_MEASUREMENTS).min()
        moving_treshold = mean_valleys - (mean_valleys - min_inv) * Utils.B_CONVEYER_DETECTION_THRESHOLD

        is_on_conveyer = 255 - values_around_point_mean < moving_treshold

        # draw debug
        cv2.rectangle(frame, (Utils.B_CONVEYER_DETECTION_POINT[0] - Utils.B_CONVEYER_DETECTION_WIDTH, Utils.B_CONVEYER_DETECTION_POINT[1] - Utils.B_CONVEYER_DETECTION_WIDTH),
                      (Utils.B_CONVEYER_DETECTION_POINT[0] + Utils.B_CONVEYER_DETECTION_WIDTH, Utils.B_CONVEYER_DETECTION_POINT[1] + Utils.B_CONVEYER_DETECTION_WIDTH), (0, 255, 0) if is_on_conveyer else (0, 0, 255), 2)

        return is_on_conveyer and len(Utils.B_CONVEYER_DETECTION_MEASUREMENTS) == Utils.B_CONVEYER_DETECTION_MEASUREMENTS.maxlen, frame
