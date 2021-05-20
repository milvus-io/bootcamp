import threading
from concurrent.futures import ThreadPoolExecutor
from service.load import do_load


def thread_runner(thread_num, func, *args):
    executor = ThreadPoolExecutor(thread_num)
    f = executor.submit(do_load, *args)
