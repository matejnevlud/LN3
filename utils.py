import cv2
import numpy as np

from config import IS_DEBUG, W, H

SPACES_STRIP_POSITION_X = 865 - 100
SPACES_STRIP_WIDTH = 80
SPACES_STRIP_KERNEL_HEIGHT = 30
SPACES_DISTANCE = 100


class Utils:

    @staticmethod
    def warp_big_conveyer_calculate(frame):
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
    def warp_big_conveyer(frame):
        t = cv2.getTickCount()

        frame = cv2.resize(frame, (1920, 1080)) if frame.shape[1] != 1920 or frame.shape[0] != 1080 else frame
        matrix = np.array([[1.49669178e+00, 3.27336992e-01, -4.57413301e+02],
                           [-1.31601690e-02, 2.01935481e+00, -4.08513578e+02],
                           [-2.83612875e-06, 3.88347700e-04, 1.00000000e+00]])

        # use,  INTER_NEAREST
        frame = cv2.warpPerspective(frame, matrix, (W, H), flags=cv2.INTER_NEAREST)
        if IS_DEBUG:
            print(f"Warp STAT: {int((cv2.getTickCount() - t) / cv2.getTickFrequency() * 1000)} ms")
        return frame

    @staticmethod
    def warp_small_conveyer(frame):
        t = cv2.getTickCount()

        pts1 = np.float32([[990 - 50, 253], [1752, 272], [952 - 50, 953], [1761, 972]])
        pts2 = np.float32([[0, 0], [H, 0], [0, H], [H, H]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)

        frame = cv2.warpPerspective(frame, matrix, (H, H), flags=cv2.INTER_NEAREST)
        if IS_DEBUG:
            print(f"Warp DYN: {int((cv2.getTickCount() - t) / cv2.getTickFrequency() * 1000)} ms")
        return frame

    @staticmethod
    def adjust_color_temperature(image, kelvin):
        def kelvin_to_rgb(k):
            k = k / 100.0
            if k <= 66:
                r = 255
                g = 99.4708025861 * np.log(k) - 161.1195681661
                b = 138.5177312231 * np.log(k - 10) - 305.0447927307 if k > 19 else 0
            else:
                r = 329.698727446 * ((k - 60) ** -0.1332047592)
                g = 288.1221695283 * ((k - 60) ** -0.0755148492)
                b = 255
            return np.clip([r, g, b], 0, 255)

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
        # frame *= np.array([0.86750843, 0.90113407, 1.]) RGB
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
        # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.erode(mask, kernel, iterations=3)
        mask = cv2.dilate(mask, kernel, iterations=3)
        # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
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

        frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))  # resize to 50% of original
        jpeg_50_quality = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])[1].tobytes()
        if q and not q.full():
            q.put(jpeg_50_quality)
