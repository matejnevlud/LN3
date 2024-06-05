IS_DEBUG = False

IS_B = False

W = 1080 if IS_B else 1920
H = 1080

NUMBER_OF_REGIONS = 8 if IS_B else 18

STRIP_X_POSITION = W // 2

HORIZONTAL_STRIP_Y_START = 510
HORIZONTAL_STRIP_HEIGHT = 220  # two times of nudle
HORIZONTAL_STRIP_HEIGHT = 100  # two times of nudle
HORIZONTAL_STRIP_Y_END = HORIZONTAL_STRIP_Y_START + HORIZONTAL_STRIP_HEIGHT

MOVING_COUNT_THRESHOLD = 200
MOVING_MEAN_THRESHOLD = 0.8

DETECTION_COOLDOWN = 1.8 if IS_B else 1.3

DETECTION_BASIC_RADIUS = 27

DETECTION_CIRCLE_RADIUS = 50
DETECTION_CIRCLE_X_OFFSETS = {
    0: 35,
    1: 30,
    2: 23,
    3: 17,
    4: 12,
    5: 5,
    6: 0,
    7: 0,
}
DETECTION_CIRCLE_PADDING = 10
DETECTION_CIRCLE_PADDING_Y_TOP = 30
DETECTION_CIRCLE_PADDING_Y_BOTTOM = 25
DETECTION_CIRCLE_SENSITIVITY = 0.7

DETECTION_SQUARE_X_OFFSETS = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    6: 0,
    7: 0,
    8: 0,
    9: 0,
    10: 0,
    11: 0,
    12: 5,
    13: 5,
    14: 5,
    15: 5,
    16: 5,
    17: 5
}

DETECTION_SQUARE_PADDING = 15
DETECTION_SQUARE_PADDING_Y_TOP = 35
DETECTION_SQUARE_PADDING_Y_BOTTOM = 15
DETECTION_SQUARE_SENSITIVITY = 0.65
