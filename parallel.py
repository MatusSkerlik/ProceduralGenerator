import multiprocessing
import signal
import time
from collections import deque
from threading import Thread

FPS = 60
THREAD_TIME = (10 ** 9) / 60
last_tick = time.monotonic_ns()
worker_queue = deque()
worker_running = True


class Worker(Thread):

    def run(self) -> None:
        global last_tick, worker_running

        while worker_running:
            if len(worker_queue) > 0:
                current_tick = time.monotonic_ns()
                if (current_tick - last_tick) < THREAD_TIME:
                    task, success, error = worker_queue.pop()
                    try:
                        success(task())
                    except Exception as e:
                        error(e)
                else:
                    # pass control to main thread
                    last_tick = time.monotonic_ns()
                    time.sleep(0)

            else:
                # pass control to main thread
                last_tick = time.monotonic_ns()
                time.sleep(0)


worker = None
process_pool = None


def to_thread(func, success_callback, error_callback):
    global worker_queue, worker

    if worker is None:
        worker = Worker()
        worker.start()
    worker_queue.append((func, success_callback, error_callback))


def to_process(func, success_callback, error_callback):
    global process_pool

    if process_pool is None:
        process_pool = multiprocessing.Pool(processes=4)
    process_pool.apply_async(func, callback=success_callback, error_callback=error_callback)


def exit_program(*args):
    global worker_running, process_pool
    worker_running = False
    if process_pool is not None:
        process_pool.terminate()
        process_pool.join()
    exit(0)


signal.signal(signal.SIGINT, exit_program)
signal.signal(signal.SIGTERM, exit_program)
