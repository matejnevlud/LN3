import time

import matplotlib.pyplot as plt
from matplotlib import colors
from collections import deque
import cv2
import numpy as np

from camera import ThreadedCamera, NormalCamera
from database import Database
from detection_point import DetectionPoint, DetectionCircle, DetectionSquare, detectCircleConveyer
from movement_detector import MovementDetector
from space_counter import SpaceCounter
from utils import Utils
from config import IS_DEBUG, MOVING_COUNT_THRESHOLD, MOVING_MEAN_THRESHOLD, HORIZONTAL_STRIP_Y_START, IS_B, DETECTION_CIRCLE_X_OFFSETS, DETECTION_CIRCLE_RADIUS


def run_main_process_b(manager_memory=None, last_frames_queue=None):
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('frame', cv2.WND_PROP_TOPMOST, 1)

    try:
        import RPi.GPIO as GPIO
        print("RUNNING ON RPI")
        cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        src = 'rtsp://admin:Manzes1997@192.168.1.64:554/ISAPI/Streaming/Channels/101'
    except ImportError:
        src = 'recordings/small_normal.mp4'
        src = 'rtsp://admin:Manzes1997@bicodigital.a.pinggy.link:18627/ISAPI/Streaming/Channels/101'
        src = 'recordings/out_09_53.mp4'
        pass

    threaded_camera = ThreadedCamera(src)
    db = Database()

    movement_detector = MovementDetector(db)
    conveyer_counter = SpaceCounter(strip_pos_x=35, strip_width=25, strip_kernel_height=90, strip_size_factor=1, strip_distance=200)
    last_measured_conveyer = None

    DETECTION_LINE_Y = 520

    frametime_deque = deque(maxlen=30)
    while True:
        try:
            t_main = cv2.getTickCount()

            frame = threaded_camera.read_frame()

            if frame is None:
                debug_frame = np.zeros((720, 1280, 3), np.uint8)
                debug_frame = cv2.putText(debug_frame, "Missing frame " + time.strftime("%Y-%m-%d %H:%M:%S"), (80, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                cv2.imshow('frame', debug_frame)
                cv2.waitKey(5)
                Utils.add_frame_to_queue(debug_frame, last_frames_queue)
                continue

            # ? warp frame to get rid of perspective
            frame = Utils.warp_b(frame)

            # remove top 300 px
            frame = frame[150:, :]

            # ? take square of regions height from the center of the region
            movement_detector.determine_movement(frame)

            conveyers = conveyer_counter.update(frame)

            conveyer_on_detection = [conveyer for conveyer in conveyers if (conveyer.h is not None and (conveyer.y < DETECTION_LINE_Y < conveyer.y + conveyer.h))]
            conveyer_on_detection = conveyer_on_detection[0] if conveyer_on_detection else None
            should_measure = conveyer_on_detection is not None and movement_detector.is_moving and conveyer_on_detection.get_h() is not None

            if should_measure:

                # ? take horizontal strip of frame between conveyers, first conveyer is y start, second is y end, take full width
                region = frame[conveyer_on_detection.y:conveyer_on_detection.y + conveyer_on_detection.h, :]

                region = Utils.adjust_color_temperature_8000_fast(region)

                # ? apply blurring for better color detection
                # region = apply_blur(region)

                # ? create mask for noodles, close and open it to remove noise
                noodles_mask = Utils.threshold_noodles(region)

                # ? detect contours and draw convex hull
                noodles_mask, region = Utils.contour_noodles(noodles_mask, region)

                detections = detectCircleConveyer(noodles_mask, 8, conveyer_on_detection)
                conveyer_on_detection.add_detections(detections)

                if last_measured_conveyer and last_measured_conveyer.uuid != conveyer_on_detection.uuid:
                    last_measured_conveyer.saved_measurements = True
                    db.save_conveyer_measurements(last_measured_conveyer)

                last_measured_conveyer = conveyer_on_detection

                # paste region onto frame at position 0, 0
                frame[conveyer_on_detection.y:region.shape[0] + conveyer_on_detection.y, 0:region.shape[1]] = region

            try:
                for i in range(len(conveyers)):
                    conveyer = conveyers[i]
                    # draw rectangle around conveyer
                    cv2.line(frame, (0, conveyer.y), (frame.shape[1], conveyer.y), conveyer.get_color(), 20)
                    if conveyer.h:
                        cv2.rectangle(frame, (0, conveyer.y), (frame.shape[1], conveyer.y + conveyer.h), conveyer.get_color(), 10)

                    if len(conveyer.measurements) > 0:
                        avg_detection = conveyer.get_avg_detection()
                        for i in range(len(avg_detection)):
                            x = frame.shape[1] // len(avg_detection) * i + frame.shape[1] // len(avg_detection) // 2 + DETECTION_CIRCLE_X_OFFSETS[i]
                            y = conveyer.y + conveyer.h // 2
                            cv2.circle(frame, (x, y), 10, (0, 255, 0) if avg_detection[i] else (0, 0, 255), -1)
                            cv2.circle(frame, (x, y), DETECTION_CIRCLE_RADIUS, (0, 255, 0) if avg_detection[i] else (0, 0, 255), 4)


            except Exception as e:
                pass

            cv2.line(frame, (0, DETECTION_LINE_Y), (frame.shape[1], DETECTION_LINE_Y), (0, 0, 255) if True else (0, 255, 255), 10)
            frametime_deque.append(int((cv2.getTickCount() - t_main) / cv2.getTickFrequency() * 1000))
            cv2.putText(frame, f"FPS: {int(1000 / np.mean(frametime_deque))}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"IS MOVING {movement_detector.white_pixels}" if movement_detector.is_moving else f"NOT MOVING {movement_detector.white_pixels}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if movement_detector.is_moving else (0, 0, 255), 2)
            print(f"FPS: {int(1000 / np.mean(frametime_deque))}    ", end='\r')
            cv2.imshow('frame', frame)

            Utils.add_frame_to_queue(frame, last_frames_queue)

            key = cv2.waitKey(threaded_camera.FPS_MS // 2) & 0xFF
            if key == ord('q'):
                exit(0)

        except Exception as e:
            print(e)
            pass
            break

    cv2.waitKey(10)
    cv2.destroyAllWindows()
    cv2.waitKey(10)

    print("exiting main process")
    if manager_memory:
        manager_memory['main_process_running'] = False
    return 0


if __name__ == '__main__':
    run_main_process_b()
