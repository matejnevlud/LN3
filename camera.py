from threading import Thread
import cv2
import time

class NormalCamera(object):
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 0)
        self.FPS = 30
        self.SLEEP_SECONDS = 1 / (self.FPS * 2)
        self.FPS_MS = int(1000 / self.FPS)


    def read_frame(self):
        if self.capture.isOpened():
            (self.status, self.frame) = self.capture.read()
            return self.frame
        else:
            return None




class ThreadedCamera(object):
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 0)

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

    def read_frame(self):
        return self.frame
