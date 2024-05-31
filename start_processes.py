import time
from multiprocessing import Manager, Queue, Process

from main import run_main_process
from server import run_uvicorn_process


IS_A = False

def start_program():
    manager = Manager()
    manager_memory = manager.dict()
    LAST_FRAMES_QUEUE = Queue(maxsize=1)

    manager_memory['main_process_running'] = True
    manager_memory['server_process_running'] = True

    main_process = Process(target=run_main_process, args=(manager_memory, LAST_FRAMES_QUEUE, IS_A,))
    main_process.daemon = True
    main_process.start()

    server_process = Process(target=run_uvicorn_process, args=(manager_memory, LAST_FRAMES_QUEUE,))
    server_process.daemon = True
    server_process.start()

    while True:
        time.sleep(1)
        is_main_process_alive = main_process.is_alive() and manager_memory['main_process_running']
        is_server_process_alive = server_process.is_alive() and manager_memory['server_process_running']

        print("Main process (" + str(main_process.pid) + ") running: " + str(is_main_process_alive), "     Server process (" + str(server_process.pid) + ") running: " + str(is_server_process_alive))
        if not is_main_process_alive or not is_server_process_alive:
            print("exiting")
            break

    main_process.terminate()
    server_process.terminate()
    print("terminating all processes in 5 sec ...")

    time.sleep(5)
    main_process.kill()
    server_process.kill()
    print("killed all processes")
    print("bye")


if __name__ == '__main__':
    start_program()
