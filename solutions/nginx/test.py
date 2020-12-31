#!/usr/bin/python3

import threading
import time
from milvus import Milvus, DataType

exitFlag = 0

class myThread (threading.Thread):
    def __init__(self, threadID, name, counter):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter
    def run(self):
        print ("开始线程：" + self.name)
        client = Milvus(host='localhost', port='19585')
        print_time(self.name, self.counter, 7)
        print ("退出线程：" + self.name)

def print_time(threadName, delay, counter):
    while counter:
        if exitFlag:
            threadName.exit()
       # time.sleep(delay)
        print ("%s: %s" % (threadName, time.ctime(time.time())))
        counter -= 1

# 创建新线程
thread1 = myThread(1, "Thread-1", 1)
thread2 = myThread(2, "Thread-2", 2)
#thread3 = myThread(3, "Thread-3", 3)

# 开启新线程
thread1.start()
thread2.start()
#thread3.start()
thread1.join()
thread2.join()
#thread3.join()
print ("退出主线程")
