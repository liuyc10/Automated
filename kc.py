from threading import Thread

import numpy as np

from calculation_tools import distance_P2Ps, cal_central_point, get_line, get_parallel_line, get_cross_pt, distance_P2P, \
    distance_P2L
from camera import Camera
import cv2 as cv

from coordinatequeue import CoordinateQueue


class KeystoneCorrection(Thread):

    def __init__(self, average=3, **kwargs):
        super(KeystoneCorrection, self).__init__()

        self.v = Camera(**kwargs)
        self.v.all_pos_reset()
        self.__cal_flag = False
        self.fuc = None
        self.fuc_w_draw = None
        self.gray = None
        self.blur_gray = None
        self.threshold_screen = None

        self.average = average
        self.corners_queue = CoordinateQueue(average)
        self.cross_queue = CoordinateQueue(average)

        self.screen_box = None
        self.cross = None

        self.corners_avg = None
        self.cross_avg = None

        self.extend_cross = None
        self.extend_rac_lines = None
        self.cross_length = None
        self.org_corners_length = None
        self.cal_corners_length = None
        self.v_tolerance_list = None
        self.h_tolerance_list = None

        self.h = self.v.height()
        self.w = self.v.width()

        self.resolution = (int(self.h), int(self.w))

        self.GAUSS = 3
        self.THRESHOLD1_SCREEN = 80
        self.THRESHOLD2_SCREEN = 200

        self.THRESHOLD1_CROSS = 140
        self.THRESHOLD2_CROSS = 250

        self.THRESHOLD1_CANNY = 200
        self.THRESHOLD2_CANNY = 250

        self.front = cv.FONT_HERSHEY_SIMPLEX
        self.scale = 0.5
        self.color = (255, 255, 255)
        self.thickness = 2

        cv.namedWindow('capture', cv.WINDOW_NORMAL)

        self.DEBUG = True

    def show_img(self, img=None, name='img'):
        cv.namedWindow(name, cv.WINDOW_NORMAL)
        if img is not None:
            cv.imshow(name, img)
        else:
            cv.imshow(name, self.fuc_w_draw)
        cv.waitKey(10)

    def draw_text(self, text, coordinate):
        cv.putText(self.fuc_w_draw, text, coordinate.astype(int), self.front, self.scale, self.color, self.thickness)

    def draw_side_length(self, corners, side_length, offset=-50):
        new_coordinate = self.__set_offset(corners, offset)
        top_left, top_right, bottom_right, bottom_left = new_coordinate
        top = (top_left + top_right) / 2
        bottom = (bottom_left + bottom_right) / 2
        left = (top_left + bottom_left) / 2
        right = (top_right + bottom_right) / 2
        for length, coordinate in zip(side_length, [top, bottom, left, right]):
            if length:
                self.draw_text(str(round(length, 2)), coordinate)
            else:
                self.draw_text('unknown', coordinate)

    def draw_tolerance(self, offset=-50):
        x_offset_coordinate = self.__set_offset(self.corners_avg, offset=offset, axis=0)
        for length, coordinate in zip(self.v_tolerance_list, x_offset_coordinate):
            if length:
                self.draw_text(str(round(length, 2)), coordinate)
            else:
                self.draw_text('unknown', coordinate)
        y_offset_coordinate = self.__set_offset(self.corners_avg, offset=offset, axis=1)
        for length, coordinate in zip(self.h_tolerance_list, y_offset_coordinate):
            if length:
                self.draw_text(str(round(length, 2)), coordinate)
            else:
                self.draw_text('unknown', coordinate)

    def draw_data(self):
        cv.drawContours(self.fuc_w_draw, [self.corners_avg.astype(int)], -1, (0, 255, 0), 1)
        cv.drawContours(self.fuc_w_draw, [self.cross_avg.astype(int)], -1, (0, 0, 255), 1)
        cv.drawContours(self.fuc_w_draw, [self.extend_cross.astype(int)], -1, (255, 255, 0), 1)
        self.draw_side_length(self.cross_avg, self.cross_length)
        self.draw_side_length(self.corners_avg, self.org_corners_length)

        # tb_tolerance, lr_tolerance = kc.cal_tolerance(corners_length)
        self.draw_side_length(self.corners_avg, self.cal_corners_length, -150)
        self.draw_tolerance()
        cv.imshow('capture', self.fuc_w_draw)

    def __set_offset(self, box, offset=0,
                     axis=2):  # axis: 0: x axis outside; 1: y axis outside; 2: x and y axis outside
        if axis == 0:
            return box + [[offset, 0],
                          [-offset, 0],
                          [-offset, 0],
                          [offset, 0]]
        elif axis == 1:
            return box + [[0, offset],
                          [0, offset],
                          [0, -offset],
                          [0, -offset]]
        elif axis == 2:
            return box + [[offset, offset],
                          [-offset, offset],
                          [-offset, -offset],
                          [offset, -offset]]
        else:
            return None

    def __mask(self, offset=10):
        mask = np.ones(self.resolution)
        mask = mask * 255
        box_w_offset = self.__set_offset(self.screen_box, offset, axis=2)
        mask = cv.fillConvexPoly(mask, box_w_offset, 0)
        return np.uint8(mask)

    def __raw_data(self):
        self.__img()
        self.screen_box = self.__edges()

        if self.screen_box is not None:
            self.cross = self.__cross()

        if self.cross is not None:
            self.__cal_flag = True
        else:
            self.__cal_flag = False

    def __img(self):
        rat, self.fuc = self.v.read()
        self.fuc_w_draw = self.fuc.copy()
        self.gray = cv.cvtColor(self.fuc.copy(), cv.COLOR_BGR2GRAY)

    def __preprocess(self, mask=None):
        if mask is not None:  # for cross
            gray_w_mask = cv.bitwise_or(self.gray, mask)
            r, threshold_screen = cv.threshold(gray_w_mask, self.THRESHOLD1_CROSS, self.THRESHOLD2_CROSS,
                                               cv.THRESH_BINARY_INV)
            if self.DEBUG:
                self.show_img(threshold_screen, 'preprocess_cross')
        else:  # for screen
            blur_gray = cv.GaussianBlur(self.gray, (self.GAUSS, self.GAUSS), 0)
            r, threshold_screen = cv.threshold(blur_gray, self.THRESHOLD1_SCREEN, self.THRESHOLD2_SCREEN,
                                               cv.THRESH_BINARY)
            if self.DEBUG:
                self.show_img(threshold_screen, 'preprocess_screen')

        blur_screen = cv.GaussianBlur(threshold_screen, (self.GAUSS, self.GAUSS), 0)
        return cv.Canny(blur_screen, threshold1=self.THRESHOLD1_CANNY,
                        threshold2=self.THRESHOLD2_CANNY)

    def __edges(self):

        canny_screen = self.__preprocess()
        contours_screen, _ = cv.findContours(canny_screen, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours_flat_screen = []
        for contour in contours_screen:
            shape = contour.shape
            contours_flat_screen.extend(contour.reshape(shape[0], 2))

        if len(contours_flat_screen) != 0:
            approx_screen = cv.approxPolyDP(np.asarray(contours_flat_screen), 10, True)
            r_approx = approx_screen.reshape(approx_screen.shape[0], 2)
            cor_top_left = r_approx[np.asarray(distance_P2Ps((0, 0), r_approx)).argmin()]
            cor_bottom_left = r_approx[np.asarray(distance_P2Ps((0, self.h), r_approx)).argmin()]
            cor_top_right = r_approx[np.asarray(distance_P2Ps((self.w, 0), r_approx)).argmin()]
            cor_bottom_right = r_approx[np.asarray(distance_P2Ps((self.w, self.h), r_approx)).argmin()]

            box = np.array([cor_top_left, cor_top_right, cor_bottom_right, cor_bottom_left])
            if self.DEBUG:
                cv.drawContours(self.fuc_w_draw, [box], -1, (255, 0, 0), 3)
                self.show_img(self.fuc_w_draw, 'edges')
            return np.array([cor_top_left, cor_top_right, cor_bottom_right, cor_bottom_left])
        else:
            return None

    def __cross(self):
        cross = []
        mask = self.__mask()
        canny_screen = self.__preprocess(mask)
        contours_screen, _ = cv.findContours(canny_screen, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if self.DEBUG:
            cv.drawContours(self.fuc_w_draw, contours_screen, -1, (0, 255, 0), 1)
        if len(contours_screen) < 4:
            return None
        for contour in contours_screen:
            c_x, c_y = cal_central_point(contour.reshape(contour.shape[0], 2))
            if c_x is None:
                continue
            cross.append([c_x, c_y])
            if self.DEBUG:
                cv.circle(self.fuc_w_draw, (round(c_x), round(c_y)), 1, (0, 0, 255), -1)
        if self.DEBUG:
            self.show_img(self.fuc_w_draw, 'cross')

        if len(cross) < 4:
            return None

        cor_top_left = cross[np.asarray(distance_P2Ps((0, 0), cross)).argmin()]
        cor_bottom_left = cross[np.asarray(distance_P2Ps((0, self.h), cross)).argmin()]
        cor_top_right = cross[np.asarray(distance_P2Ps((self.w, 0), cross)).argmin()]
        cor_bottom_right = cross[np.asarray(distance_P2Ps((self.w, self.h), cross)).argmin()]
        cor = np.array([cor_top_left, cor_top_right, cor_bottom_right, cor_bottom_left])

        return cor

    def __cross_rec(self):
        cor_top_left, cor_top_right, cor_bottom_right, cor_bottom_left = self.cross_avg
        top_line = get_line(cor_top_left, cor_top_right)
        bottom_line = get_line(cor_bottom_left, cor_bottom_right)
        left_line = get_line(cor_top_left, cor_bottom_left)
        right_line = get_line(cor_top_right, cor_bottom_right)
        return np.asarray([top_line, bottom_line, left_line, right_line])

    def __extend_rec(self, base_pts=0):
        # base_pts: 0: middle of line
        #           1: top left and bottom right
        #           2: top right and bottom left

        top_left, top_right, bottom_right, bottom_left = self.corners_avg
        lines = self.__cross_rec()
        top_line, bottom_line, left_line, right_line = lines

        if base_pts == 0:
            top_pt = (top_left + top_right) / 2
            bottom_pt = (bottom_left + bottom_right) / 2
            left_pt = (top_left + bottom_left) / 2
            right_pt = (top_right + bottom_right) / 2
        elif base_pts == 1:
            top_pt = top_left
            bottom_pt = bottom_right
            left_pt = top_left
            right_pt = bottom_right
        else:
            top_pt = top_right
            bottom_pt = bottom_left
            left_pt = bottom_left
            right_pt = top_right
        top_line_e = get_parallel_line(top_pt, top_line)
        left_line_e = get_parallel_line(left_pt, left_line)
        bottom_line_e = get_parallel_line(bottom_pt, bottom_line)
        right_line_e = get_parallel_line(right_pt, right_line)
        cor_top_left = get_cross_pt(top_line_e, left_line_e)
        cor_top_right = get_cross_pt(top_line_e, right_line_e)
        cor_bottom_right = get_cross_pt(bottom_line_e, right_line_e)
        cor_bottom_left = get_cross_pt(bottom_line_e, left_line_e)
        extend_rec_cor = np.asarray([cor_top_left, cor_top_right, cor_bottom_right, cor_bottom_left])
        extend_rec_line = np.asarray([top_line_e, bottom_line_e, left_line_e, right_line_e])
        return extend_rec_cor, extend_rec_line

    def __cal_length(self, pts):
        top_left, top_right, bottom_right, bottom_left = pts
        top_length = distance_P2P(top_left, top_right)
        bottom_length = distance_P2P(bottom_left, bottom_right)
        left_length = distance_P2P(top_left, bottom_left)
        right_length = distance_P2P(top_right, bottom_right)

        return np.asarray([top_length, bottom_length, left_length, right_length])

    def __length_calibration(self):
        # [top_length, bottom_length, left_length, right_length]
        corners_length = self.__cal_length(self.corners_avg)  # [top_length, bottom_length, left_length, right_length]
        cross_length = self.__cal_length(self.cross_avg)  # [top_length, bottom_length, left_length, right_length]

        cal_top_corner_length = cross_length[0] * corners_length[1] / cross_length[1]
        cal_left_corner_length = cross_length[2] * corners_length[3] / cross_length[3]

        return [cal_top_corner_length, corners_length[1], cal_left_corner_length, corners_length[3]]

    def __tolerance(self):
        top_left, top_right, bottom_right, bottom_left = self.corners_avg
        top_line_e, bottom_line_e, left_line_e, right_line_e = self.extend_rac_lines
        self.v_tolerance_list = [distance_P2L(top_left, top_line_e),
                                 distance_P2L(top_right, top_line_e),
                                 distance_P2L(bottom_right, bottom_line_e),
                                 distance_P2L(bottom_left, bottom_line_e)]
        self.h_tolerance_list = [distance_P2L(top_left, left_line_e),
                                 distance_P2L(top_right, right_line_e),
                                 distance_P2L(bottom_right, right_line_e),
                                 distance_P2L(bottom_left, left_line_e)]

    def __process(self):

        self.corners_queue.push(self.screen_box)
        self.cross_queue.push(self.cross)
        self.corners_avg = self.corners_queue.average()
        self.cross_avg = self.cross_queue.average()

        self.extend_cross, self.extend_rac_lines = self.__extend_rec()
        self.cross_length = self.__cal_length(self.cross_avg)
        self.org_corners_length = self.__cal_length(self.corners_avg)
        self.cal_corners_length = self.__length_calibration()
        self.__tolerance()

    def tolerance(self, draw=False):
        if self.v.isOpened():
            self.__raw_data()
            if self.__cal_flag:
                self.__process()
                self.__tolerance()
                if draw:
                    self.draw_data()
        else:
            print('No camera found, please check your device.')

    def run(self):

        try:
            while self.v.isOpened():
                self.__raw_data()
                if self.__cal_flag:
                    self.__process()
                    self.__tolerance()
                    self.draw_data()

                if cv.waitKey(1) & 0xFF == ord('q'):
                    self.v.release()
                    break
        finally:
            self.v.release()
