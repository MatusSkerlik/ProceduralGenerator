import multiprocessing
import random
import string
import time
from functools import partial
from threading import Thread, Lock
from typing import Dict

FPS = 25
THREAD_TIME = (10 ** 9) / FPS
_last_tick = time.monotonic_ns()

_to_thread_queue = list()

_err_tasks = list()
_err_tasks_lock = Lock()
_succ_tasks = list()
_succ_tasks_lock = Lock()

# functionality of 'after token' calls of this module
# each token has its own list of tasks witch waits for task completion
# thread version
_thread_lock = Lock()  # lock for MainThread and AfterTokenThread
_after_token_tasks_to_thread: Dict[str, list] = dict()

# functionality of 'after token' calls of this module
# each token has its own list of tasks witch waits for task completion
# process version
_process_lock = Lock()  # lock for MainThread and AfterTokenThread
_after_token_tasks_to_process: Dict[str, list] = dict()


def get_token():
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(10))


_to_thread_worker_running = True


class _ToThreadWorker(Thread):
    name = 'ToThreadWorker'

    def run(self) -> None:
        global _last_tick

        while _to_thread_worker_running:
            current_tick = time.monotonic_ns()
            while (current_tick - _last_tick) < THREAD_TIME:
                if len(_to_thread_queue) > 0:
                    task, success, error = _to_thread_queue.pop()
                    try:
                        success(task())
                    except Exception as e:
                        error(e)
                    current_tick = time.monotonic_ns()
                else:
                    break
                # pass control to main thread
            time.sleep(1 / FPS)
            _last_tick = time.monotonic_ns()


_after_token_worker_running = True


class _AfterTokenWorker(Thread):
    name = 'AfterTokenWorker'

    def run(self) -> None:
        while _after_token_worker_running:
            with _succ_tasks_lock, _thread_lock, _process_lock:
                for task_token, result in _succ_tasks:
                    if task_token in _after_token_tasks_to_thread:
                        q = _after_token_tasks_to_thread[task_token]
                        while len(q) > 0:
                            func, succ, err, token = q.pop()
                            to_thread(partial(func, result), succ, err, token)
                        del _after_token_tasks_to_thread[task_token]

                    if task_token in _after_token_tasks_to_process:
                        q = _after_token_tasks_to_process[task_token]
                        while len(q) > 0:
                            func, succ, err, token = q.pop()
                            to_process(partial(func, result), succ, err, token)
                        del _after_token_tasks_to_process[task_token]

            with _err_tasks_lock, _thread_lock, _process_lock:
                for task_token, error in _err_tasks:
                    if task_token in _after_token_tasks_to_thread:
                        q = _after_token_tasks_to_thread[task_token]
                        while len(q) > 0:
                            func, succ, err, token = q.pop()
                            err(error)
                        del _after_token_tasks_to_thread[task_token]

                    if task_token in _after_token_tasks_to_process:
                        q = _after_token_tasks_to_process[task_token]
                        while len(q) > 0:
                            func, succ, err, token = q.pop()
                            err(error)
                        del _after_token_tasks_to_process[task_token]
            time.sleep(1 / FPS)


_to_thread_worker = None
_after_token_worker = None


def to_thread(func, success_callback, error_callback, token=get_token()):
    """
    Run func in thread and calls appropriate callback after finish (within ToThreadWorker)

    Also saves result of func into array for to_thread_after func calls
    """
    global _to_thread_queue

    # THIS WILL BE CALLED FROM POOL THREAD WORKER NUM 3
    def after_success(result):
        with _succ_tasks_lock:
            _succ_tasks.append((token, result))
        success_callback(result)

    # THIS WILL BE CALLED FROM POOL THREAD WORKER NUM 3
    def after_err(error):
        with _err_tasks_lock:
            _err_tasks.append((token, error))
        error_callback(error)

    # THIS IS CALLED IN MAIN THREAD
    _to_thread_queue.append((func, after_success, after_err))


def to_thread_after(func, success_callback, error_callback, wait_token, token=get_token()):
    data = (func, success_callback, error_callback, token)
    if wait_token in _after_token_tasks_to_thread:
        q = _after_token_tasks_to_thread[wait_token]
        q.append(data)
    else:
        q = list()
        q.append(data)
        _after_token_tasks_to_thread[wait_token] = q


_process_pool = None


def to_process(func, success_callback, error_callback, token=get_token()):
    """
    Run func in process and calls appropriate callback after finish (within process pool thread)

    Also saves result of func into array for to_process_after func calls
    """
    global _process_pool

    # THIS WILL BE CALLED FROM POOL THREAD WORKER NUM 3
    def after_success(result):
        with _succ_tasks_lock:
            _succ_tasks.append((token, result))
        success_callback(result)

    # THIS WILL BE CALLED FROM POOL THREAD WORKER NUM 3
    def after_err(error):
        with _err_tasks_lock:
            _err_tasks.append((token, error))
        error_callback(error)

    _process_pool.apply_async(func, callback=after_success, error_callback=after_err)


def to_process_after(func, success_callback, error_callback, wait_token, token=get_token()):
    data = (func, success_callback, error_callback, token)
    if wait_token in _after_token_tasks_to_process:
        q = _after_token_tasks_to_process[wait_token]
        q.append(data)
    else:
        q = list()
        q.append(data)
        _after_token_tasks_to_process[wait_token] = q


def init():
    global _process_pool, _to_thread_worker, _after_token_worker
    if _process_pool is None:
        _process_pool = multiprocessing.Pool()
    if _to_thread_worker is None:
        global _to_thread_worker_running
        _to_thread_worker_running = True

        _to_thread_worker = _ToThreadWorker()
        _to_thread_worker.daemon = True
        _to_thread_worker.start()
    if _after_token_worker is None:
        global _after_token_worker_running
        _after_token_worker_running = True

        _after_token_worker = _AfterTokenWorker()
        _after_token_worker.deamon = True
        _after_token_worker.start()


def clean():
    global _process_pool, _to_thread_worker, _after_token_worker
    if _process_pool is not None:
        _process_pool.terminate()
        _process_pool.join()
    if _to_thread_worker is not None:
        global _to_thread_worker_running
        _to_thread_worker_running = False

        _to_thread_worker.join()
    if _after_token_worker is not None:
        global _after_token_worker_running
        _after_token_worker_running = False

        _after_token_worker.join()
