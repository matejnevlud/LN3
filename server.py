import time
from multiprocessing import Queue

import numpy as np
import uvicorn
from fastapi import FastAPI
from starlette.responses import StreamingResponse, Response

app = FastAPI()

LAST_FRAMES_QUEUE = Queue(maxsize=1)


@app.get("/video_feed")
async def video_feed():
    def streamer():
        try:
            while True:
                if LAST_FRAMES_QUEUE.empty():
                    yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                           b'\r\n')
                else:
                    frame = LAST_FRAMES_QUEUE.get()
                    yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                           bytearray(frame) + b'\r\n')

                time.sleep(0.1)
        except GeneratorExit:
            print("cancelled")

    return StreamingResponse(streamer(), media_type="multipart/x-mixed-replace;boundary=frame")


@app.get("/")
async def last_frame():
    if LAST_FRAMES_QUEUE.empty():
        return Response(content=b"", media_type="image/jpeg")
    return Response(content=LAST_FRAMES_QUEUE.get(), media_type="image/jpeg")


def run_uvicorn_process(manager_memory, frame_queue):
    global LAST_FRAMES_QUEUE
    LAST_FRAMES_QUEUE = frame_queue

    try:
        uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
    except Exception as e:
        print(e)
        manager_memory['server_process_running'] = False
        print("server process crashed")
        pass
    print("server process stopped")
    manager_memory['server_process_running'] = False
    return
