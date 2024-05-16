from threading import Thread
import cv2, time

class ThreadedCamera(object):
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.FPS = 1/100
        self.FPS_MS = int(self.FPS * 1000)
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
            time.sleep(self.FPS)


if __name__ == '__main__':
    src = 'rtsp://admin:Manzes1997@192.168.1.64:554/ISAPI/Streaming/Channels/101'
    src = 'rtsp://admin:Manzes1997@4.tcp.eu.ngrok.io:19401/ISAPI/Streaming/Channels/101'
    #src = 'rtsp://admin:Manzes1997,@192.168.1.64:554/media/video2'
    threaded_camera = ThreadedCamera(src)
    while True:
        try:
            cv2.imshow('frame', threaded_camera.frame)
            cv2.waitKey(threaded_camera.FPS_MS)
        except AttributeError:
            pass

#ffmpeg -rtsp_transport tcp -i "rtsp://admin:Manzes1997@192.168.1.64:554/ISAPI/Streaming/Channels/103" -c copy /home/matejnevlud/app/output.mp4