import os
import cv2
import time
import threading

global thFrame
global ret
thFrame = None
ret = False

class ImageGrabber(threading.Thread):
    def __init__(self, cap ):
        global thFrame
        threading.Thread.__init__(self)
        self.cap = cap
        _,thFrame = self.cap.read()

    def run(self):
        global thFrame
        global ret
        while True:
            ret,img = self.cap.read()
            if ret != True: break
            thFrame = img
            #time.sleep(0.2)
        self.cap.release()

class DataAquisition:
    def __init__(self,mode):
        self.mode = mode
        self.read = None

    def initialize(self,input_):
        global thFrame
        global ret

        if self.mode == "folder":
            def read(input_):
                list_img = [os.path.join(input_,i) for i in sorted(os.listdir(input_))]
                for arq in list_img:
                    frame = cv2.imread(arq)
                    yield True, frame
                yield False, None
            self.read = read(input_)

        elif self.mode == "video":
            print(input_)
            cap = cv2.VideoCapture(input_)
            grabber = ImageGrabber(cap)
            grabber.start()

            def read(input_):
                ret = True
                while True:
                    if ret != True: break
                    yield ret, thFrame
                yield False, None
            self.read = read(input_)

        elif self.mode == "webcam":
            print('found webcam')
            cap = cv2.VideoCapture(0)
            grabber = ImageGrabber(cap)
            grabber.start()

            def read(input_):
                print('def webcam module')
                ret = True
                while True:
                    print('while webcam module')
                    print(ret)
                    if ret != True: break
                    yield ret, thFrame
                yield False, None


            self.read = read(input_)
        return cap

    def get(self):
        return next(self.read)
