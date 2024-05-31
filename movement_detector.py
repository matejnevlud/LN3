from collections import deque
import cv2
import numpy as np

MOVING_COUNT_THRESHOLD = 200
MOVING_MEAN_THRESHOLD = 0.8




class MovementDetector:
    def __init__(self, db):
        self.bg_subtractor = cv2.bgsegm.createBackgroundSubtractorCNT(minPixelStability=1, useHistory=False, maxPixelStability=2, isParallel=True)
        self.white_pixels = 0
        self.was_moving_deque = deque(maxlen=200)
        self.is_moving = False
        self.db = db

    def determine_movement(self, frame):
        square = frame[frame.shape[0] // 2 - 50:frame.shape[0] // 2 + 50, frame.shape[1] // 2 - 50:frame.shape[1] // 2 + 50]
        fgmask = self.bg_subtractor.apply(square)

        self.white_pixels = cv2.countNonZero(fgmask)
        self.was_moving_deque.append(self.white_pixels > MOVING_COUNT_THRESHOLD)

        # did change from moving to not moving or vice versa
        new_is_moving = np.mean(self.was_moving_deque) > MOVING_MEAN_THRESHOLD and len(self.was_moving_deque) == self.was_moving_deque.maxlen
        if new_is_moving != self.is_moving:
            self.is_moving = new_is_moving
            self.db.add_movement(1 if new_is_moving else 0)
