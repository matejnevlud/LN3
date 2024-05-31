import time
from multiprocessing import Manager, Queue, Process

from main import run_main_process
from server import run_uvicorn_process


IS_A = True


def start_main_process(manager_memory, LAST_FRAMES_QUEUE):
    main_process = Process(target=run_main_process, args=(manager_memory, LAST_FRAMES_QUEUE, IS_A,))
    main_process.daemon = True
    main_process.start()
    manager_memory['main_process_running'] = True

    return main_process


def start_server_process(manager_memory, LAST_FRAMES_QUEUE):
    server_process = Process(target=run_uvicorn_process, args=(manager_memory, LAST_FRAMES_QUEUE,))
    server_process.daemon = True
    server_process.start()
    manager_memory['server_process_running'] = True

    return server_process

def kill_all_processes(main_process, server_process):
    main_process.terminate()
    server_process.terminate()
    print("terminating all processes in 5 sec ...")

    time.sleep(5)
    main_process.kill()
    server_process.kill()

def start_program():
    manager = Manager()
    manager_memory = manager.dict()
    LAST_FRAMES_QUEUE = Queue(maxsize=1)

    main_process = start_main_process(manager_memory, LAST_FRAMES_QUEUE)
    server_process = start_server_process(manager_memory, LAST_FRAMES_QUEUE)

    while True:
        time.sleep(5)
        is_main_process_alive = main_process.is_alive() and manager_memory['main_process_running']
        is_server_process_alive = server_process.is_alive() and manager_memory['server_process_running']
        if not is_main_process_alive or not is_server_process_alive:
            kill_all_processes(main_process, server_process)

            main_process = start_main_process(manager_memory, LAST_FRAMES_QUEUE)
            server_process = start_server_process(manager_memory, LAST_FRAMES_QUEUE)
            print("restarting all processes")


    print("killed all processes")
    print("bye")


if __name__ == '__main__':
    start_program()
