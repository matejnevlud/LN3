import time
import traceback
from collections import deque

import cv2
import numpy as np

from camera import ThreadedCamera
from config import DETECTION_CIRCLE_X_OFFSETS, DETECTION_CIRCLE_RADIUS, DETECTION_SQUARE_X_OFFSETS
from database import Database
from detection_point import detect_small_conveyer, detect_big_conveyer
from movement_detector import MovementDetector
from space_counter import SpaceCounter
from utils import Utils


def init_space_counter(IS_A):
    if IS_A:
        conveyer_counter = SpaceCounter(strip_pos_x=865 + 390, strip_width=80, strip_kernel_height=50, strip_size_factor=1, strip_distance=100)
    else:
        conveyer_counter = SpaceCounter(strip_pos_x=35, strip_width=25, strip_kernel_height=90, strip_size_factor=1, strip_distance=200)
    return conveyer_counter


def draw_detections(frame, conveyers, IS_A):
    for i in range(len(conveyers)):
        conveyer = conveyers[i]
        # draw rectangle around conveyer
        cv2.line(frame, (0, conveyer.y), (frame.shape[1], conveyer.y), conveyer.get_color(), 20)

        if conveyer.h is None:
            cv2.putText(frame, "NEW CONVEYER", (10, conveyer.y + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, conveyer.get_color(), 2)
            continue

        cv2.rectangle(frame, (0, conveyer.y), (frame.shape[1], conveyer.y + conveyer.h), conveyer.get_color(), 10)

        if len(conveyer.measurements) > 0:
            avg_detection = conveyer.get_avg_detection()
            for i in range(len(avg_detection)):
                if IS_A:
                    x = frame.shape[1] // len(avg_detection) * i + frame.shape[1] // len(avg_detection) // 2 + DETECTION_SQUARE_X_OFFSETS[i]
                    y = conveyer.y + conveyer.h // 2
                    w = frame.shape[1] // 18 - 2 * 10
                    cv2.circle(frame, (x, y), 10, (0, 255, 0) if avg_detection[i] else (0, 0, 255), -1)
                    cv2.rectangle(frame, (x - w // 2, y - w // 2), (x + w // 2, y + w // 2), (0, 255, 0) if avg_detection[i] else (0, 0, 255), 3)
                else:
                    x = frame.shape[1] // len(avg_detection) * i + frame.shape[1] // len(avg_detection) // 2 + DETECTION_CIRCLE_X_OFFSETS[i]
                    y = conveyer.y + conveyer.h // 2
                    cv2.circle(frame, (x, y), 10, (0, 255, 0) if avg_detection[i] else (0, 0, 255), -1)
                    cv2.circle(frame, (x, y), DETECTION_CIRCLE_RADIUS, (0, 255, 0) if avg_detection[i] else (0, 0, 255), 4)


def draw_debug_info(frame, line_y, frametime_deque, movement_detector):
    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 0, 255) if True else (0, 255, 255), 10)

    if len(frametime_deque) > 0:
        cv2.putText(frame, f"FPS: {int(1000 / np.mean(frametime_deque))}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"IS MOVING {movement_detector.white_pixels}" if movement_detector.is_moving else f"NOT MOVING {movement_detector.white_pixels}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if movement_detector.is_moving else (0, 0, 255), 2)


def run_main_process(manager_memory=None, last_frames_queue=None, IS_A=True):
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('frame', cv2.WND_PROP_TOPMOST, 1)

    try:
        import RPi.GPIO as GPIO
        print("RUNNING ON RPI")
        cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        src = 'rtsp://admin:Manzes1997@192.168.1.64:554/ISAPI/Streaming/Channels/101'
    except ImportError:
        src = 'rtsp://admin:Manzes1997@bicodigital.a.pinggy.link:18627/ISAPI/Streaming/Channels/101'
        src = 'recordings/out_09_53.mp4' if IS_A else 'recordings/small_opa.mp4'
        pass

    threaded_camera = ThreadedCamera(src)
    db = Database()

    movement_detector = MovementDetector(db)
    conveyer_counter = init_space_counter(IS_A)
    last_measured_conveyer = None

    detection_line_y = 820 if IS_A else 520

    frametime_deque = deque(maxlen=30)
    while True:
        try:
            t_main = cv2.getTickCount()

            frame = threaded_camera.read_frame()

            # ? check if frame is None and display error message
            if frame is None:
                debug_frame = np.zeros((720, 1280 if IS_A else 720, 3), np.uint8)
                debug_frame = cv2.putText(debug_frame, "NO VIDEO INPUT", (80, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                debug_frame = cv2.putText(debug_frame, time.strftime("%Y-%m-%d %H:%M:%S"), (80, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                cv2.imshow('frame', debug_frame)
                Utils.add_frame_to_queue(debug_frame, last_frames_queue)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
                continue

            # ? warp frame to get rid of perspective
            frame = Utils.warp_big_conveyer(frame) if IS_A else Utils.warp_small_conveyer(frame)

            # ? remove top 150 px if B
            frame = frame if IS_A else frame[150:, :]

            # ? take square of regions height from the center of the region
            movement_detector.determine_movement(frame)

            # ? update conveyer counter state
            conveyers = conveyer_counter.update(frame)

            # ? determine if is moving and if conveyer is on detection line so we can measure presence of noodles
            conveyer_on_detection = [conveyer for conveyer in conveyers if (conveyer.h is not None and (conveyer.y < detection_line_y < conveyer.y + conveyer.h))]
            conveyer_on_detection = conveyer_on_detection[0] if conveyer_on_detection else None
            should_measure = conveyer_on_detection is not None and movement_detector.is_moving and conveyer_on_detection.get_h() is not None

            if should_measure:

                # ? take horizontal strip of frame between conveyers, first conveyer is y start, second is y end, take full width
                region = frame[conveyer_on_detection.y:conveyer_on_detection.y + conveyer_on_detection.h, :]

                # ? adjust color temperature to 8000 if B
                region = region if IS_A else Utils.adjust_color_temperature_8000_fast(region)

                # ? create mask for noodles, close and open it to remove noise
                noodles_mask = Utils.threshold_noodles(region)

                # ? detect contours and draw convex hull
                noodles_mask, region = Utils.contour_noodles(noodles_mask, region)

                detections = detect_big_conveyer(noodles_mask) if IS_A else detect_small_conveyer(noodles_mask, 8, conveyer_on_detection)
                conveyer_on_detection.add_detections(detections)

                if last_measured_conveyer and last_measured_conveyer.uuid != conveyer_on_detection.uuid:
                    last_measured_conveyer.saved_measurements = True
                    db.save_conveyer_measurements(last_measured_conveyer)

                last_measured_conveyer = conveyer_on_detection

                # paste region onto frame at position 0, 0
                frame[conveyer_on_detection.y:region.shape[0] + conveyer_on_detection.y, 0:region.shape[1]] = region

            # ? draw measured conveyers on frame
            draw_detections(frame, conveyers, IS_A)

            # ? draw debug info
            draw_debug_info(frame, detection_line_y, frametime_deque, movement_detector)

            # ? add frame to multiprocessing queue
            Utils.add_frame_to_queue(frame, last_frames_queue)

            # ? calculate FPS
            frametime_deque.append(int((cv2.getTickCount() - t_main) / cv2.getTickFrequency() * 1000))

            # ? show frame
            cv2.imshow('frame', frame)
            key = cv2.waitKey(threaded_camera.FPS_MS // 2) & 0xFF
            if key == ord('q'):
                break

        except Exception as e:
            traceback.print_exc()
            # write to log file traceback
            with open('error_log.txt', 'a') as f:
                f.write(traceback.format_exc())
            break

    cv2.waitKey(10)
    cv2.destroyAllWindows()
    cv2.waitKey(10)

    print("exiting main process")
    if manager_memory:
        manager_memory['main_process_running'] = False
    return 0


if __name__ == '__main__':
    run_main_process()
