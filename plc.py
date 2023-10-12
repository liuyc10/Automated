import csv
from datetime import datetime
from threading import Thread

import cv2 as cv
import numpy as np
from scipy.spatial.distance import pdist

import calculation_tools
from camera import Camera


class PLC(Thread):

    def __init__(self, **kwargs):
        super(PLC, self).__init__()
        self.coordinates = None
        self.DEBUG = False
        self.v = Camera(**kwargs)

        self.height = self.v.height()
        self.width = self.v.width()

        self.current_frame = None
        self.skip_frame_no = 0

        self.timestamp = None
        cv.namedWindow('frame', cv.WINDOW_NORMAL)
        cv.createButton('start', self.start_test)

    def start_test(self, x, s):
        raise NotImplementedError

    def time(self, full=True):
        t = datetime.now()
        if full:
            return '{}{}{}_{}{}{}{}'.format(t.year, t.month, t.day,
                                            t.hour, t.minute, t.second,
                                            t.microsecond if len(str(t.microsecond)) == 6 else '0' + str(
                                                t.microsecond))
        else:
            return '{}{}{}_{}_{}_{}'.format(t.year, t.month, t.day, t.hour, t.minute, t.second)

    def add_TimeStamp(self):
        cv.putText(self.current_frame, self.timestamp, (100, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        if len(self.coordinates) != 0:
            for number, coordinate in self.coordinates.items():
                cv.putText(self.current_frame, str(number), (coordinate + [-50, 10]).astype(int),
                           cv.FONT_HERSHEY_SIMPLEX, 1,
                           (255, 0, 255), 2)

    def searching(self):
        image = self.current_frame
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        r, threshold_screen = cv.threshold(gray.copy(), 200, 255, cv.THRESH_BINARY)
        if self.DEBUG:
            self.show_img(threshold_screen)
        blur_screen = cv.GaussianBlur(threshold_screen, (11, 11), 0)
        canny_screen = cv.Canny(blur_screen, threshold1=200, threshold2=250)
        contours_screen, _ = cv.findContours(canny_screen, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # cv.drawContours(image, contours_screen, -1, (0, 0, 255), 10)
        return contours_screen

    def raw_data(self):
        rat, self.current_frame = self.v.read()
        if self.current_frame is None:
            return None
        self.timestamp = self.time()
        circles = self.searching()
        return np.asarray(
            [calculation_tools.cal_central_point(circle.reshape(circle.shape[0], 2)) for circle in circles])

    def show_img(self, img, name='img'):
        cv.namedWindow(name, cv.WINDOW_NORMAL)
        cv.imshow(name, img)
        cv.waitKey(10)

    def release(self):
        self.v.release()

    def initial_setup(self):
        raise NotImplementedError

    def current_status(self):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError
