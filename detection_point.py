import cv2

from config import W, DETECTION_SQUARE_X_OFFSETS, DETECTION_CIRCLE_SENSITIVITY, DETECTION_SQUARE_SENSITIVITY, DETECTION_SQUARE_PADDING, DETECTION_CIRCLE_X_OFFSETS, DETECTION_SQUARE_PADDING_Y_TOP, \
    DETECTION_SQUARE_PADDING_Y_BOTTOM, DETECTION_CIRCLE_RADIUS


def detect_big_conveyer(mask):
    detections = []
    for i in range(0, 18):
        x = int(W / 18 * i) + DETECTION_SQUARE_PADDING + DETECTION_SQUARE_X_OFFSETS[i]
        y = DETECTION_SQUARE_PADDING_Y_TOP
        w = W // 18 - 2 * DETECTION_SQUARE_PADDING
        test_area = w * w

        h = mask.shape[0] - DETECTION_SQUARE_PADDING_Y_TOP - DETECTION_SQUARE_PADDING_Y_BOTTOM
        focused_region = mask[y:y + h, x:x + w]

        detected = cv2.countNonZero(focused_region) > DETECTION_SQUARE_SENSITIVITY * test_area

        detections.append(detected)

    return detections


def detect_small_conveyer(mask, no_of_regions, conveyer):
    detections = []
    for i in range(0, no_of_regions):
        x = mask.shape[1] // no_of_regions * i + mask.shape[1] // no_of_regions // 2 + DETECTION_CIRCLE_X_OFFSETS[i]
        y = mask.shape[0] // 2
        r = DETECTION_CIRCLE_RADIUS

        # get region of interest from mask
        focused_region = mask[y - r:y + r, x - r:x + r]
        test_area = 3.1415 * r * r

        detected = cv2.countNonZero(focused_region) > DETECTION_CIRCLE_SENSITIVITY * test_area

        detections.append(detected)

    return detections
