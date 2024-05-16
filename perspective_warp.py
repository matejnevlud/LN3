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
    src = 'rtsp://admin:Manzes1997@bicodigital.a.pinggy.link:18627/ISAPI/Streaming/Channels/101'
    #src = 'rtsp://admin:Manzes1997,@192.168.1.64:554/media/video2'
    threaded_camera = ThreadedCamera(src)

    #named window
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('frame', cv2.WND_PROP_TOPMOST, 1)


    while True:
        try:

            # when clicked on window, print coordinates
            def click_event(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    print(x, y)
            cv2.setMouseCallback('frame', click_event)


            cv2.imshow('frame', threaded_camera.frame)
            cv2.waitKey(0)
        except AttributeError:
            pass

#ffmpeg -rtsp_transport tcp -i "rtsp://admin:Manzes1997@192.168.1.64:554/ISAPI/Streaming/Channels/103" -c copy /home/matejnevlud/app/output.mp4