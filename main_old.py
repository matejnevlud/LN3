import matplotlib.pyplot as plt
from matplotlib import colors
from collections import deque
import cv2
import numpy as np

from camera import ThreadedCamera, NormalCamera
from database import Database
from detection_point import DetectionPoint, DetectionCircle, DetectionSquare
from utils import Utils
from config import IS_DEBUG, MOVING_COUNT_THRESHOLD, MOVING_MEAN_THRESHOLD, HORIZONTAL_STRIP_Y_START, IS_B

frametime_deque = deque(maxlen=30)

if __name__ == '__main__':
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('frame', cv2.WND_PROP_TOPMOST, 1)



    def apply_blur(frame_bgr):
        frame_bgr = cv2.GaussianBlur(frame_bgr, (11, 11), 0)
        #frame_bgr = cv2.medianBlur(frame_bgr, 11)
        #frame_bgr = horizontal_gauss(frame_bgr)
        #frame_bgr = cv2.bilateralFilter(frame_bgr, 17, 128, 50)
        return frame_bgr

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


    mog2 = cv2.createBackgroundSubtractorMOG2()
    mog2.setHistory(100)
    mog2.setVarThreshold(16)
    mog2.setDetectShadows(False)
    white_pixels = 0
    was_moving_deque = deque(maxlen=200)
    is_moving = False

    def determine_movement(region):
        square = region[0:region.shape[0], int(region.shape[1]/2 - region.shape[0]/2):int(region.shape[1]/2 + region.shape[0]/2)]
        fgmask = mog2.apply(square)
        fgmask = cv2.resize(fgmask, (50, 50))
        if IS_DEBUG:
            cv2.imshow('MOG movement mask', fgmask)
        # count white pixels
        global white_pixels
        white_pixels = cv2.countNonZero(fgmask)
        was_moving_deque.append(white_pixels > MOVING_COUNT_THRESHOLD)

        # did change from moving to not moving or vice versa
        global is_moving
        new_is_moving = np.mean(was_moving_deque) > MOVING_MEAN_THRESHOLD and len(was_moving_deque) == was_moving_deque.maxlen
        if new_is_moving != is_moving:
            is_moving = new_is_moving
            db.add_movement(new_is_moving)


    def estimate_color_temperature(img):
        # Convert the img to the Lab color space
        lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        # Calculate the average of the a and b channels
        avg_a = np.mean(lab_img[:, :, 1])
        avg_b = np.mean(lab_img[:, :, 2])

        # Use a simple model to convert a and b values to color temperature
        # This model is an approximation and may not be highly accurate
        # The formula can vary depending on empirical data
        tmp = 1000 * (1.0 + avg_b - avg_a)

        return tmp

    reference = cv2.imread('/Users/matejnevlud/github/LN3/captures/14_05/20240514_144830.jpg')
    src = 'rtsp://admin:Manzes1997@rnqwc-93-99-154-195.a.free.pinggy.link:43979/ISAPI/Streaming/Channels/101'
    src = 'rtsp://admin:Manzes1997@bicodigital.a.pinggy.link:18627/ISAPI/Streaming/Channels/101'
    #src = 'recordings/out_locked.mp4'
    src = 'recordings/out_10_03.mp4'


    # determine if running on raspberry pi
    try:
        import RPi.GPIO as GPIO
        print("RUNNING ON RPI")
        src = 'rtsp://admin:Manzes1997@192.168.1.64:554/ISAPI/Streaming/Channels/101'
    except ImportError:
        pass

    threaded_camera = ThreadedCamera(src)
    db = Database()

    Detection_model = DetectionCircle if IS_B else DetectionSquare
    detection_points = Detection_model.init_points(db)

    while True:
        try:
            t = cv2.getTickCount()
            #!frame is BGR !!!
            frame = threaded_camera.read_frame()

            #? warp frame to get rid of perspective
            frame = Utils.warp_b(frame) if IS_B else Utils.warp_conveyer_calculate(frame)


            vert = Utils.take_strip(frame)
            #vert = Utils.horizontal_gauss(vert)
            vert = Utils.detect_ridges(vert)
            if IS_DEBUG:
                cv2.imshow('vert', vert)



            #temp = estimate_color_temperature(frame)
            #frame = adjust_color_temperature(frame, 10000)

            #? take horizontal strip of frame
            region = Utils.take_horizontal_strip(frame)

            #? apply temp color if B
            region = Utils.adjust_color_temperature(region, 8000) if IS_B else region

            #? apply blurring for better color detection
            region = apply_blur(region)

            #? take square of regions height from the center of the region
            determine_movement(region)



            #? create mask for noodles, close and open it to remove noise
            noodles_mask = threshold_noodles(region)

            #? detect contours and draw convex hull
            noodles_mask, region = contour_noodles(noodles_mask, region)



            #? update state for detection points
            for point in detection_points:
                point.update(noodles_mask, is_moving)
                point.draw(region, is_moving)







            # paste region onto frame at position 0, 0
            frame[HORIZONTAL_STRIP_Y_START:region.shape[0] + HORIZONTAL_STRIP_Y_START, 0:region.shape[1]] = region
            cv2.rectangle(frame, (0, HORIZONTAL_STRIP_Y_START), (frame.shape[1], region.shape[0] + HORIZONTAL_STRIP_Y_START), (0, 0, 255), 3)

            #TODO : ONLY DEBUG DRAWING
            if IS_DEBUG:
                whole_frame_noodles_mask, frame = contour_noodles(threshold_noodles(apply_blur(frame)), frame)
                # detect regions in vert, draw them
                contours, _ = cv2.findContours(vert, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    hull = cv2.convexHull(contour)
                    cv2.drawContours(frame, [hull], 0, (0, 255, 0), -1)



            frametime_deque.append(int((cv2.getTickCount() - t) / cv2.getTickFrequency() * 1000))
            cv2.putText(frame, f"FPS: {int(1000 / np.mean(frametime_deque))}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"IS MOVING {white_pixels}" if is_moving else f"NOT MOVING {white_pixels}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if is_moving else (0, 0, 255), 2)
            #cv2.putText(frame, f"TEMP: {temp}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if is_moving else (0, 0, 255), 2)



            print(f"FPS: {int(1000 / np.mean(frametime_deque))}    ", end='\r')
            cv2.imshow('frame', frame)






            key = cv2.waitKey(threaded_camera.FPS_MS // 2) & 0xFF


            if key == ord('d'):
                IS_DEBUG = not IS_DEBUG
            #if h is pressed, show histogram of the frame in matplotlib
            if key == ord('h'):
                # histogram of HSV only hue

                small_frame = cv2.resize(frame, (640, 480))
                hsv = cv2.cvtColor(small_frame, cv2.COLOR_RGB2HSV)
                h, s, v = cv2.split(hsv)

                plt.figure(figsize=(16, 10))
                axis = plt.axes(projection="3d")

                pixel_colors = small_frame.reshape((np.shape(small_frame)[0]*np.shape(small_frame)[1], 3))
                norm = colors.Normalize(vmin=-1.,vmax=1.)
                norm.autoscale(pixel_colors)
                pixel_colors = norm(pixel_colors).tolist()


                axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
                axis.set_xlabel("Hue")
                axis.set_ylabel("Saturation")
                axis.set_zlabel("Value")
                # flip camera view elevation 23, azimuth 81, roll 0
                axis.view_init(elev=23, azim=120)

                plt.show()

            # wait key 30 ms, if q, exit
            if key == ord('q'):
                exit(0)


        except Exception as e:
            print(e)
            pass
            exit(-1)

exit(0)



