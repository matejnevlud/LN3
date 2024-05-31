import hashlib
import uuid

import cv2
import numpy as np
from scipy.signal import find_peaks

from config import H

SPACES_STRIP_POSITION_X = 865 + 390
SPACES_STRIP_WIDTH = 80
SPACES_STRIP_KERNEL_HEIGHT = 30
SPACES_DISTANCE = 100


def hash_string_to_3d_vector(input_string: str):
    # Generate a SHA-256 hash of the input string
    hash_object = hashlib.sha256(input_string.encode())
    hash_hex = hash_object.hexdigest()

    # Convert the hash to an integer
    hash_int = int(hash_hex, 16)

    # Extract 3 values from the hash integer and map them to the range 0-255
    vector = []
    for i in range(3):
        # Shift the hash integer to get a different part for each value
        value = (hash_int >> (i * 8)) & 0xFF
        vector.append(value)

    return vector


class Conveyer:
    def __init__(self, y):
        self.y = y
        self.h = None

        # generate random uuid
        self.uuid = uuid.uuid4()
        self.measurements = []

        self.saved_measurements = False

    def set_uuid(self, u):
        self.uuid = u

    def set_h(self, h):
        self.h = h

    def get_h(self):
        return self.h

    def get_color(self):
        return hash_string_to_3d_vector(str(self.uuid))

    def add_detections(self, detections):
        self.measurements.append(detections)

    def get_avg_detection(self):
        return [np.mean(x) for x in zip(*self.measurements)]


SPACE_GUTTER = 50


class SpaceCounter:

    def __init__(self, strip_pos_x=None, strip_width=None, strip_kernel_height=None, strip_distance=None, strip_size_factor=None):
        self.last_spaces = np.array([])

        self.strip_pos_x = strip_pos_x or SPACES_STRIP_POSITION_X
        self.strip_width = strip_width or SPACES_STRIP_WIDTH
        self.strip_kernel_height = strip_kernel_height or SPACES_STRIP_KERNEL_HEIGHT
        self.strip_distance = strip_distance or SPACES_DISTANCE
        self.strip_size_factor = strip_size_factor or 1

        pass

    def find_spaces(self, frame_bgr):
        frame_lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l_channel = frame_lab[:, :, 0]
        strip = l_channel[0:frame_bgr.shape[0], self.strip_pos_x:self.strip_pos_x + self.strip_width]

        kernel = np.ones((self.strip_kernel_height, 1), np.float32) / self.strip_kernel_height
        filtered_strip = cv2.filter2D(strip, -1, kernel, borderType=cv2.BORDER_CONSTANT)
        values = filtered_strip[:, 0]

        values = np.array(values)
        # invert values
        values = 255 - values
        # mean
        mean = values.mean()

        spaces, _ = find_peaks(values, distance=self.strip_distance, height=mean)

        # for each peak, traverse left and right to find stem of peak
        spaces_obj = []
        for peak in spaces:
            stem = values[peak] - 3
            left = peak
            right = peak
            while left > 0 and values[left] > stem:
                left -= 5
            while right < len(values) and values[right] > stem:
                right += 5

            new_peak = (left + right) // 2

            spaces_obj.append(Conveyer(new_peak))

        # set h as distances between peaks
        for i in range(0, len(spaces_obj) - 1):
            # apply factor to h
            spaces_obj[i].set_h(abs(spaces_obj[i + 1].y - spaces_obj[i].y) * self.strip_size_factor)

        return spaces_obj

    def update(self, frame_bgr):
        new_spaces = self.find_spaces(frame_bgr)

        if len(self.last_spaces) == 0:
            self.last_spaces = new_spaces
            return new_spaces

        # remove spaces that are too close to the top
        new_spaces = [space for space in new_spaces if space.y > SPACE_GUTTER]

        # remove spaces that are too close to the bottom
        new_spaces = [space for space in new_spaces if space.y < H - SPACE_GUTTER]

        # compare new spaces with last spaces, using gutter find overlaping spaces
        for new_space in new_spaces:
            did_find_space = False

            for last_space in self.last_spaces:
                if abs(new_space.y - last_space.y) < SPACE_GUTTER:
                    new_space.set_uuid(last_space.uuid)
                    new_space.measurements = last_space.measurements
                    self.last_spaces = [space for space in self.last_spaces if space.uuid != last_space.uuid]
                    did_find_space = True
                    break

            if not did_find_space:
                # print(f"New space found: {new_space.uuid}")
                pass

        self.last_spaces = new_spaces

        return new_spaces
