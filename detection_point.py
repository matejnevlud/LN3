import cv2
import numpy as np
import time
from config import W, DETECTION_BASIC_RADIUS, DETECTION_COOLDOWN, IS_DEBUG, DETECTION_CIRCLE_PADDING, DETECTION_SQUARE_X_OFFSETS, DETECTION_CIRCLE_SENSITIVITY, DETECTION_SQUARE_SENSITIVITY, DETECTION_SQUARE_PADDING, NUMBER_OF_REGIONS, DETECTION_CIRCLE_X_OFFSETS, DETECTION_SQUARE_PADDING_Y_TOP, \
    DETECTION_SQUARE_PADDING_Y_BOTTOM, DETECTION_CIRCLE_PADDING_Y_TOP, DETECTION_CIRCLE_PADDING_Y_BOTTOM, DETECTION_CIRCLE_RADIUS


class DetectionPoint:

    #def static method
    @staticmethod
    def init_points(database):
        detection_points = []
        for i in range(0, NUMBER_OF_REGIONS):
            x = int(W / NUMBER_OF_REGIONS * i + (W / NUMBER_OF_REGIONS / 2))
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

    def update(self, mask, is_moving):
        self.mask = mask
        # test if circle with radius 10 is in beige mask with center in point
        circle_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.circle(circle_mask, (self.x, self.y), DETECTION_BASIC_RADIUS, 255, -1)
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

    def draw(self, frame, is_moving):
        if not is_moving:
            cv2.circle(frame, (self.x, self.y), DETECTION_BASIC_RADIUS, (0, 140, 255), -1)
        elif self.is_detected:
            cv2.circle(frame, (self.x, self.y), DETECTION_BASIC_RADIUS, (0, 255, 0), -1)
        else:
            cv2.circle(frame, (self.x, self.y), DETECTION_BASIC_RADIUS, (0, 0, 255), -1)

        # draw count under the point
        cv2.putText(frame, str(self.counter), (self.x - 15, self.y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def __str__(self):
        return f"({self.x}, {self.y})"

    def __repr__(self):
        return self.__str__()



class DetectionCircle:
    #def static method
    @staticmethod
    def init_points(database):
        detection_points = []
        for i in range(0, 8):
            x = int(W / 8 * i) + DETECTION_CIRCLE_PADDING + DETECTION_CIRCLE_X_OFFSETS[i]
            y = 50 + DETECTION_CIRCLE_PADDING
            d = W//8 - 2 * DETECTION_CIRCLE_PADDING
            r = d // 2
            detection_points.append(DetectionCircle(i, x, y, r, database))
        return detection_points

    def __init__(self, region_id, x, y, r, database):
        self.counter = 0
        self.x = x
        self.y = y
        self.r = r
        self.database = database

        self.region_id = region_id
        self.is_detected = False
        self.detected_time = 0
        self.mask = None


        circle = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r+r, r+r)) * 255
        self.test_mask = np.zeros((r+r, r+r), dtype=np.uint8)

        #self.test_mask[r//2:r+r//2, r//2:r+r//2] = circle
        self.test_mask = circle

        self.test_area = cv2.countNonZero(self.test_mask)



    def update(self, mask, is_moving):
        focused_region = mask[self.y:self.y+self.r+self.r, self.x:self.x+self.r+self.r]

        # test if focused region has same dimensions as test mask

        anded_mask = cv2.bitwise_and(focused_region, self.test_mask)

        if IS_DEBUG:
            self.anded_percent = cv2.countNonZero(anded_mask) / self.test_area
        detected = cv2.countNonZero(anded_mask) > DETECTION_CIRCLE_SENSITIVITY * self.test_area

        # check if is already detected
        if self.is_detected:
            if time.time() - self.detected_time > DETECTION_COOLDOWN:
                self.is_detected = False
            return

        if not is_moving:
            return

        # check if is detected
        if detected:
            self.is_detected = True
            self.detected_time = time.time()
            self.counter += 1
            self.database.add_detection(self.region_id)
        else:
            self.is_detected = False


    def draw(self, frame, is_moving):
        if not is_moving:
            cv2.circle(frame, (self.x + self.r, self.y + self.r), self.r, (0, 140, 255), -1)

        elif self.is_detected:
            cv2.circle(frame, (self.x + self.r, self.y + self.r), self.r, (0, 255, 0), -1)
        else:
            cv2.circle(frame, (self.x + self.r, self.y + self.r), self.r, (0, 0, 255), -1)

        # draw count under the point
        cv2.putText(frame, str(self.counter), (self.x - 15, self.y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if IS_DEBUG:
            cv2.putText(frame, f"{self.anded_percent:.2f}", (self.x + 10, self.y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)





class DetectionSquare:
    #def static method
    @staticmethod
    def init_points(database):
        detection_points = []
        for i in range(0, NUMBER_OF_REGIONS):
            x = int(W / NUMBER_OF_REGIONS * i) + DETECTION_SQUARE_PADDING + DETECTION_SQUARE_X_OFFSETS[i]
            y = DETECTION_SQUARE_PADDING_Y_TOP
            w = W//NUMBER_OF_REGIONS - 2 * DETECTION_SQUARE_PADDING
            detection_points.append(DetectionSquare(i, x, y, w, database))
        return detection_points

    def __init__(self, region_id, x, y, w, database):
        self.counter = 0
        self.x = x
        self.y = y
        self.w = w
        self.database = database

        self.region_id = region_id
        self.is_detected = False
        self.detected_time = 0
        self.mask = None

        self.test_area = w * w

    def update(self, mask, is_moving):
        h = mask.shape[0] - DETECTION_SQUARE_PADDING_Y_TOP - DETECTION_SQUARE_PADDING_Y_BOTTOM
        focused_region = mask[self.y:self.y+h, self.x:self.x+self.w]

        if IS_DEBUG:
            self.anded_percent = cv2.countNonZero(focused_region) / self.test_area
        detected = cv2.countNonZero(focused_region) > DETECTION_SQUARE_SENSITIVITY * self.test_area

        # check if is already detected
        if self.is_detected:
            if time.time() - self.detected_time > DETECTION_COOLDOWN:
                self.is_detected = False
            return

        if not is_moving:
            return

        # check if is detected
        if detected:
            self.is_detected = True
            self.detected_time = time.time()
            self.counter += 1
            self.database.add_detection(self.region_id)
        else:
            self.is_detected = False


    def draw(self, frame, is_moving):
        h = frame.shape[0] - DETECTION_SQUARE_PADDING_Y_TOP - DETECTION_SQUARE_PADDING_Y_BOTTOM

        if not is_moving:
            cv2.rectangle(frame, (self.x, self.y), (self.x+self.w, self.y + h), (0, 140, 255), -1)
        elif self.is_detected:
            cv2.rectangle(frame, (self.x, self.y), (self.x+self.w, self.y+ h), (0, 255, 0), -1)
        else:
            cv2.rectangle(frame, (self.x, self.y), (self.x+self.w, self.y + h), (0, 0, 255), -1)

        # draw count under the point
        cv2.putText(frame, str(self.counter), (self.x - 15, self.y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if IS_DEBUG:
            cv2.putText(frame, f"{self.anded_percent:.2f}", (self.x + 10, self.y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)






def detectSquareConveyer(mask):
    detections = []
    for i in range(0, 18):
        x = int(W / 18 * i) + DETECTION_SQUARE_PADDING + DETECTION_SQUARE_X_OFFSETS[i]
        y = DETECTION_SQUARE_PADDING_Y_TOP
        w = W//18 - 2 * DETECTION_SQUARE_PADDING
        test_area = w * w

        h = mask.shape[0] - DETECTION_SQUARE_PADDING_Y_TOP - DETECTION_SQUARE_PADDING_Y_BOTTOM
        focused_region = mask[y:y+h, x:x+w]

        detected = cv2.countNonZero(focused_region) > DETECTION_SQUARE_SENSITIVITY * test_area

        detections.append(detected)

    return detections

def detectCircleConveyer(mask, no_of_regions, conveyer):
    detections = []
    for i in range(0, no_of_regions):
        x = mask.shape[1] // no_of_regions * i + mask.shape[1] // no_of_regions // 2 + DETECTION_CIRCLE_X_OFFSETS[i]
        y = mask.shape[0] // 2
        r = DETECTION_CIRCLE_RADIUS

        # get region of interest from mask
        focused_region = mask[y-r:y+r, x-r:x+r]
        test_area = 3.1415 * r * r

        detected = cv2.countNonZero(focused_region) > DETECTION_CIRCLE_SENSITIVITY * test_area

        detections.append(detected)

    return detections


class DetectionSquareConveyer:
    #def static method
    @staticmethod
    def init_points(database):
        detection_points = []
        for i in range(0, NUMBER_OF_REGIONS):
            x = int(W / NUMBER_OF_REGIONS * i) + DETECTION_SQUARE_PADDING + DETECTION_SQUARE_X_OFFSETS[i]
            y = DETECTION_SQUARE_PADDING_Y_TOP
            w = W//NUMBER_OF_REGIONS - 2 * DETECTION_SQUARE_PADDING
            detection_points.append(DetectionSquare(i, x, y, w, database))
        return detection_points

    def __init__(self, region_id, x, y, w, database):
        self.counter = 0
        self.x = x
        self.y = y
        self.w = w
        self.database = database

        self.region_id = region_id
        self.is_detected = False
        self.detected_time = 0
        self.mask = None

        self.test_area = w * w

    def detect(self, mask, is_moving):
        h = mask.shape[0] - DETECTION_SQUARE_PADDING_Y_TOP - DETECTION_SQUARE_PADDING_Y_BOTTOM
        focused_region = mask[self.y:self.y+h, self.x:self.x+self.w]

        if IS_DEBUG:
            self.anded_percent = cv2.countNonZero(focused_region) / self.test_area
        detected = cv2.countNonZero(focused_region) > DETECTION_SQUARE_SENSITIVITY * self.test_area

        # check if is already detected
        if self.is_detected:
            if time.time() - self.detected_time > DETECTION_COOLDOWN:
                self.is_detected = False
            return

        if not is_moving:
            return

        # check if is detected
        if detected:
            self.is_detected = True
            self.detected_time = time.time()
            self.counter += 1
            self.database.add_detection(self.region_id)
        else:
            self.is_detected = False


    def draw(self, frame, is_moving):
        h = frame.shape[0] - DETECTION_SQUARE_PADDING_Y_TOP - DETECTION_SQUARE_PADDING_Y_BOTTOM

        if not is_moving:
            cv2.rectangle(frame, (self.x, self.y), (self.x+self.w, self.y + h), (0, 140, 255), -1)
        elif self.is_detected:
            cv2.rectangle(frame, (self.x, self.y), (self.x+self.w, self.y+ h), (0, 255, 0), -1)
        else:
            cv2.rectangle(frame, (self.x, self.y), (self.x+self.w, self.y + h), (0, 0, 255), -1)

        # draw count under the point
        cv2.putText(frame, str(self.counter), (self.x - 15, self.y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if IS_DEBUG:
            cv2.putText(frame, f"{self.anded_percent:.2f}", (self.x + 10, self.y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


