from multiprocessing import Process, Lock
import time

class MyProcess(Process):
    def __init__(self, name, mutex=None):
        super().__init__()
        self.name = name 
        self.mutex = mutex

    def run(self):
        if self.mutex:
            self.mutex.acquire()
        print('%s 1' % self.name)
        time.sleep(1)
        print('%s 2' % self.name)
        time.sleep(1)
        print('%s 3' % self.name)
        if self.mutex:
            self.mutex.release() 


if __name__ == "__main__":
    mutex = Lock()
    for i in range(3):
        p = MyProcess("进程{}".format(i))
        # p.daemon = True
        p.start()
        # p.join()