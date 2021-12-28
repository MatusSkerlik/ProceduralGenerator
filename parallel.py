import random
import string
import threading
import time
from threading import Thread
from typing import Any, Tuple, Union, Callable

FPS = 25


class Future:
    id: str
    _result: Any
    _exception: Any
    _done: bool
    _result_mapper = staticmethod(lambda x: x)

    def __init__(self, func, succ, err):
        self.id = ''.join(random.choice(string.ascii_lowercase) for _ in range(20))
        self._after = set()
        self._done = False
        self._result = None
        self._exception = None
        self._lock = threading.Lock()

        self.func = func
        self.succ = succ
        self.err = err

    def __hash__(self):
        return hash(self.id)

    def after(self, *args: Union[Tuple['Future', ...], None], subscribe_for_result: bool = True):
        if self.done:
            return
        with self._lock:
            for arg in args:
                assert isinstance(arg, Future)
                self._after.add(After(arg, subscribe_for_result))
            return self

    def get_after(self):
        with self._lock:
            return self._after

    @property
    def done(self):
        with self._lock:
            return self._done

    @done.setter
    def done(self, value):
        assert isinstance(value, bool)
        with self._lock:
            self._done = value

    @property
    def result(self):
        if self.done:
            with self._lock:
                return self._result
        else:
            raise ValueError("Result not ready yet.")

    @result.setter
    def result(self, value):
        with self._lock:
            self._result = value

    @property
    def result_mapper(self):
        with self._lock:
            return self._result_mapper

    @result_mapper.setter
    def result_mapper(self, value):
        assert isinstance(value, Callable)
        with self._lock:
            self._result_mapper = staticmethod(value)

    @property
    def exception(self):
        if self.done:
            with self._lock:
                return self._exception
        else:
            raise ValueError("Exception not ready yet.")

    @exception.setter
    def exception(self, value):
        assert isinstance(value, Exception)
        with self._lock:
            self._exception = value


class After:

    def __init__(self, future: Future, subscribe_result: bool) -> None:
        self.future = future
        self.subscribe_result = subscribe_result


class ConcurrentExecutor:
    class Runner(Thread):

        def __init__(self):
            super().__init__()
            self.running = True

        def terminate(self):
            self.running = False

        def run(self) -> None:
            while self.running:
                with ConcurrentExecutor._lock:
                    pending = tuple(ConcurrentExecutor.pending.values())

                ready = []
                for future in pending:
                    is_ready = True
                    for after in future.get_after():
                        if not after.future.done:
                            is_ready = False
                            break
                    if is_ready:
                        ready.append(future)
                ready.reverse()

                while len(ready) > 0 and self.running:
                    future = ready.pop()
                    results = []
                    for after in future.get_after():
                        if after.subscribe_result:
                            results.append(after.future.result_mapper(after.future.result))
                    try:
                        # TODO thread pool or process pool
                        if len(results) > 0:
                            result = future.func(*results)
                        else:
                            result = future.func()
                        future.done = True
                        future.result = result
                        future.succ(result)
                        with ConcurrentExecutor._lock:
                            ConcurrentExecutor.pending.pop(future.id)
                    except Exception as exception:
                        future.done = True
                        future.exception = exception
                        future.err(exception)
                        with ConcurrentExecutor._lock:
                            ConcurrentExecutor.pending.pop(future.id)

                time.sleep(1 / FPS)

    pending = {}
    _lock = threading.Lock()
    _runner: Runner = None

    @classmethod
    def submit(cls, arg0: Union[Callable, Future], arg1: Union[Future, Callable, None],
               arg2: Union[Future, Callable, None]):
        assert isinstance(arg0, Callable) or isinstance(arg0, Future)
        if isinstance(arg0, Callable):
            echo = lambda x: print(x)
            func = arg0
            succ = echo
            err = echo
            assert isinstance(arg1, Callable) or arg1 is None
            if isinstance(arg1, Callable):
                succ = arg1
            assert isinstance(arg2, Callable) or arg2 is None
            if isinstance(arg2, Callable):
                err = arg2
            future = Future(func, succ, err)
            with ConcurrentExecutor._lock:
                cls.pending[future.id] = future
            return future
        else:
            future = arg0
            with ConcurrentExecutor._lock:
                cls.pending[future.id] = future
            return future

    @classmethod
    def run(cls):
        if cls._runner is None:
            cls._runner = ConcurrentExecutor.Runner()
            cls._runner.start()

    @classmethod
    def terminate(cls):
        if cls._runner is not None:
            cls._runner.terminate()

    @classmethod
    def join(cls):
        if cls._runner is not None:
            cls._runner.join()
