import cv2 as cv
import numpy as np
import location_detemination as ld

DEBUG = True
GAUSS = 3
THRESHOLD1 = 200
THRESHOLD2 = 250
cor_left = [0, 0]
cor_right = [1919, 0]


def show_img(img, name='img'):
    cv.namedWindow(name, cv.WINDOW_NORMAL)
    cv.imshow(name, img)
    cv.waitKey(10)


def draw_text(img, text, coordinate, front=cv.FONT_HERSHEY_SIMPLEX, scale=0.5, color=(255, 255, 255), thickness=2):
    cv.putText(img, text, coordinate.astype(int), front, scale, color, thickness)


def draw_side_length(img, corners, side_length, offset=-50):
    new_coordinate = set_offset(corners, offset)
    top_left, top_right, bottom_right, bottom_left = new_coordinate
    top = (top_left + top_right) / 2
    bottom = (bottom_left + bottom_right) / 2
    left = (top_left + bottom_left) / 2
    right = (top_right + bottom_right) / 2
    for length, coordinate in zip(side_length, [top, bottom, left, right]):
        if length:
            draw_text(img, str(round(length, 2)), coordinate)
        else:
            draw_text(img, 'unknown', coordinate)


def draw_tolerance(img, corners, extend_rac_lines, offset=-50):

    top_left, top_right, bottom_right, bottom_left = corners
    top_line_e, bottom_line_e, left_line_e, right_line_e = extend_rac_lines
    v_tolerance_list = [cal_get_perpendicular_line_length(top_left, top_line_e),
                        cal_get_perpendicular_line_length(top_right, top_line_e),
                        cal_get_perpendicular_line_length(bottom_right, bottom_line_e),
                        cal_get_perpendicular_line_length(bottom_left, bottom_line_e)]
    x_offset_coordinate = set_offset(corners, offset=offset, axis=0)
    for length, coordinate in zip(v_tolerance_list, x_offset_coordinate):
        draw_text(img, str(int(length)), coordinate)
    h_tolerance_list = [cal_get_perpendicular_line_length(top_left, left_line_e),
                        cal_get_perpendicular_line_length(top_right, right_line_e),
                        cal_get_perpendicular_line_length(bottom_right, right_line_e),
                        cal_get_perpendicular_line_length(bottom_left, left_line_e)]
    y_offset_coordinate = set_offset(corners, offset=offset, axis=1)
    for length, coordinate in zip(h_tolerance_list, y_offset_coordinate):
        if length:
            draw_text(img, str(round(length, 2)), coordinate)
        else:
            draw_text(img, 'unknown', coordinate)


def set_offset(box, offset=0, axis=2):  # axis: 1: x axis outside; 2: y axis outside; 3: x and y axis outside

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


def make_mask(box, offset=0):
    mask = np.ones((1080, 1920))
    mask = mask * 255
    box_w_offset = set_offset(box, offset, axis=2)
    mask = cv.fillConvexPoly(mask, box_w_offset, 0)
    return np.uint8(mask)


def get_line(pt0, pt1):
    x0, y0 = pt0
    x1, y1 = pt1
    return np.asarray([y1 - y0, x0 - x1, x1 * y0 - x0 * y1])


def get_perpendicular_line(pt, line):
    x, y = pt
    a, b, c = line
    return np.asarray([b, -a, a * y - b * x])


def cal_get_perpendicular_line_length(pt, line):
    perpendicular_line = get_perpendicular_line(pt, line)
    cross_pt = get_cross_pt(line, perpendicular_line)
    return distance_P2P(pt, cross_pt)


def get_parallel_line(pt, line):
    x, y = pt
    a, b, c = line
    c = -(a * x + b * y)
    return np.asarray([a, b, c])


def get_cross_rec(cross):
    cor_top_left, cor_top_right, cor_bottom_right, cor_bottom_left = cross
    top_line = get_line(cor_top_left, cor_top_right)
    bottom_line = get_line(cor_bottom_left, cor_bottom_right)
    left_line = get_line(cor_top_left, cor_bottom_left)
    right_line = get_line(cor_top_right, cor_bottom_right)
    return np.asarray([top_line, bottom_line, left_line, right_line])


def get_extend_rec(corners, cross, base_pts=0):
    # base_pts: 0: middle of line
    #           1: top left and bottom right
    #           2: top right and bottom left

    top_left, top_right, bottom_right, bottom_left = corners
    lines = get_cross_rec(cross)
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


def get_cross_pt(line0, line1):
    a0, b0, c0 = line0
    a1, b1, c1 = line1
    y = (c1 * a0 - c0 * a1) / (b0 * a1 - b1 * a0)
    x = (c1 * b0 - c0 * b1) / (a0 * b1 - a1 * b0)
    return np.asarray([x, y])


def distance_P2Ps(pt0, pts):
    return [(pt0[0] - t[0]) ** 2 + (pt0[1] - t[1]) ** 2 for t in pts]


def distance_P2P(pt0, pt1):
    return ((pt0[0] - pt1[0]) ** 2 + (pt0[1] - pt1[1]) ** 2) ** 0.5


def distance_P2L(pt, line):
    x, y = pt
    a, b, c = line
    return abs(a * x + b * y + c) / ((a ** 2 + b ** 2) ** 0.5)


def cal_area(pts):
    area = 0.0
    x0, y0 = pts[-1]
    for x1, y1 in pts:
        area += (y1 * x0 - x1 * y0)
        y0, x0 = y1, x1
    return area / 2


def cal_central_point(pts):
    if len(pts) < 3:
        return np.uint8((pts[0] + pts[-1]) / 2)
    else:
        area = cal_area(pts)

        if area == 0.0:
            return None
        else:
            xc, yc = 0.0, 0.0
            x0, y0 = pts[-1]
            for x1, y1 in pts:
                xc += ((x1 + x0) * (y1 * x0 - y0 * x1))
                yc += ((y1 + y0) * (y1 * x0 - y0 * x1))
                y0, x0 = y1, x1
            return np.asarray([round(xc / (6 * area)), round(yc / (6 * area))])


def cal_length(corners):
    top_left, top_right, bottom_right, bottom_left = corners
    top_length = distance_P2P(top_left, top_right)
    bottom_length = distance_P2P(bottom_left, bottom_right)
    left_length = distance_P2P(top_left, bottom_left)
    right_length = distance_P2P(top_right, bottom_right)

    return np.asarray([top_length, bottom_length, left_length, right_length])


def length_calibration(corners, cross):
    # [top_length, bottom_length, left_length, right_length]
    corners_length = cal_length(corners)  # [top_length, bottom_length, left_length, right_length]
    cross_length = cal_length(cross)  # [top_length, bottom_length, left_length, right_length]

    cal_top_corner_length = cross_length[0] * corners_length[1] / cross_length[1]
    cal_left_corner_length = cross_length[2] * corners_length[3] / cross_length[3]

    return [cal_top_corner_length, corners_length[1], cal_left_corner_length, corners_length[3]]


def cal_tolerance(corners_length):
    top_bottom_tolerance = corners_length[0] - corners_length[1]
    left_right_tolerance = corners_length[2] - corners_length[3]
    return top_bottom_tolerance, left_right_tolerance


def get_edges(image, mask=None):
    global GAUSS, THRESHOLD2, THRESHOLD1, DEBUG, cor_right, cor_left
    box = []
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    r, threshold_screen = cv.threshold(gray.copy(), 80, 255, cv.THRESH_BINARY)
    if DEBUG:
        show_img(threshold_screen)
    blur_screen = cv.GaussianBlur(threshold_screen, (GAUSS, GAUSS), 0)
    canny_screen = cv.Canny(blur_screen, threshold1=THRESHOLD1, threshold2=THRESHOLD2)

    contours_screen, _ = cv.findContours(canny_screen, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    contours_flat_screen = []
    for contour in contours_screen:
        shape = contour.shape
        contours_flat_screen.extend(contour.reshape(shape[0], 2))

    h, w, d = image.shape

    if len(contours_flat_screen) != 0:
        approx_screen = cv.approxPolyDP(np.asarray(contours_flat_screen), 10, True)
        r_approx = approx_screen.reshape(approx_screen.shape[0], 2)
        cor_top_left = r_approx[np.asarray(distance_P2Ps((0, 0), r_approx)).argmin()]
        cor_bottom_left = r_approx[np.asarray(distance_P2Ps((0, h), r_approx)).argmin()]
        cor_top_right = r_approx[np.asarray(distance_P2Ps((w, 0), r_approx)).argmin()]
        cor_bottom_right = r_approx[np.asarray(distance_P2Ps((w, h), r_approx)).argmin()]
        box = np.array([cor_top_left, cor_top_right, cor_bottom_right, cor_bottom_left])
        if DEBUG:
            cv.drawContours(image, [box], -1, (255, 0, 0), 3)
        return box
    else:
        return None


def get_cross(image, mask=None):
    global GAUSS, THRESHOLD2, THRESHOLD1, DEBUG, cor_right, cor_left
    cross = []
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray_w_mask = cv.bitwise_or(gray.copy(), mask)
    r, threshold_screen = cv.threshold(gray_w_mask, 100, 255, cv.THRESH_BINARY_INV)
    if DEBUG:
        show_img(threshold_screen, 'cross')
    blur_screen = cv.GaussianBlur(threshold_screen, (GAUSS, GAUSS), 0)
    canny_screen = cv.Canny(blur_screen, threshold1=THRESHOLD1, threshold2=THRESHOLD2)
    # if DEBUG:
    #    show_img(canny_screen)
    contours_screen, _ = cv.findContours(canny_screen, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if DEBUG:
        cv.drawContours(image, contours_screen, -1, (0, 255, 0), 1)
    if len(contours_screen) != 4:
        return None
    for contour in contours_screen:
        c_x, c_y = cal_central_point(contour.reshape(contour.shape[0], 2))
        if c_x is None:
            return None
        cross.append([c_x, c_y])
        if DEBUG:
            cv.circle(image, (c_x, c_y), 1, (0, 0, 255), -1)
    if DEBUG:
        show_img(image, 'none')

    h, w, d = image.shape

    cor_top_left = cross[np.asarray(distance_P2Ps((0, 0), cross)).argmin()]
    cor_bottom_left = cross[np.asarray(distance_P2Ps((0, h), cross)).argmin()]
    cor_top_right = cross[np.asarray(distance_P2Ps((w, 0), cross)).argmin()]
    cor_bottom_right = cross[np.asarray(distance_P2Ps((w, h), cross)).argmin()]
    cor = np.array([cor_top_left, cor_top_right, cor_bottom_right, cor_bottom_left])

    return cor


def use_canny(imgs):
    img = None
    gray = None
    if isinstance(imgs, list):
        img = imgs[0]
        gray = get_gray(imgs)
    else:
        img = imgs
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # gray = cv.fastNlMeansDenoising(gray, None, 15, 8, 25)
    global GAUSS, THRESHOLD2, THRESHOLD1, DEBUG, cor_right, cor_left

    if DEBUG:
        show_img(gray)
    r, threshold_screen = cv.threshold(gray.copy(), 100, 255, cv.THRESH_TOZERO)
    if DEBUG:
        show_img(threshold_screen)
    blur_screen = cv.GaussianBlur(threshold_screen, (GAUSS, GAUSS), 0)
    canny_screen = cv.Canny(blur_screen, threshold1=THRESHOLD1, threshold2=THRESHOLD2)
    revers = cv.bitwise_not(gray.copy())
    rr, threshold_ruler = cv.threshold(revers, 240, 255, cv.THRESH_TOZERO)
    if DEBUG:
        show_img(threshold_ruler, 'ruler')
    blur_ruler = cv.GaussianBlur(threshold_ruler, (GAUSS, GAUSS), 0)
    rr, threshold_ruler = cv.threshold(blur_ruler, 1, 255, cv.THRESH_BINARY)
    blur_ruler = cv.GaussianBlur(threshold_ruler, (GAUSS, GAUSS), 0)
    rr, threshold_ruler = cv.threshold(blur_ruler, 1, 255, cv.THRESH_BINARY)
    blur_ruler = cv.GaussianBlur(threshold_ruler, (GAUSS, GAUSS), 0)
    canny_ruler = cv.Canny(blur_ruler, threshold1=100, threshold2=250)

    h, w, d = img.shape

    contours_screen, _ = cv.findContours(canny_screen, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    contours_ruler, __ = cv.findContours(canny_ruler, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    contours_flat_screen = []
    for contour in contours_screen:
        shape = contour.shape
        contours_flat_screen.extend(contour.reshape(shape[0], 2))

    contours_flat_ruler = []
    for contour in contours_ruler:
        shape = contour.shape
        contours_flat_ruler.extend(contour.reshape(shape[0], 2))

    # a = contours[0].reshape(787, 2)

    # contours_list = np.asarray(contours)

    """i = 0
    r = 255
    b = 0
    for contour in contours:
        cv.drawContours(img, contour, -1, (b, 0, r), 3)
        cv.putText(img, str(i), contour[0].reshape(2), cv.FONT_HERSHEY_PLAIN, 5, (b, 0, r), 3)
        # int(len(contour)/2)
        i += 1
        r = 255 - r
        b = 255 - b"""

    # cv.drawContours(img, [approx], -1, (0, 255, 0), 3)
    cv.drawContours(img, contours_ruler, -1, (0, 0, 255), 5)

    if len(contours_flat_ruler) != 0:
        approx_ruler = cv.approxPolyDP(np.asarray(contours_flat_ruler), 1080, True)

        r_approx_ruler = approx_ruler.reshape(approx_ruler.shape[0], 2)

        for cor in r_approx_ruler:
            if cor[0] == 0:  # and cor[1] > 700:
                cor_left = cor
            elif cor[0] == 1919:  # and cor[1] > 700:
                cor_right = cor
        cv.line(img, cor_left, cor_right, (0, 255, 0), 5)

    if len(contours_flat_screen) != 0:
        approx_screen = cv.approxPolyDP(np.asarray(contours_flat_screen), 10, True)
        r_approx = approx_screen.reshape(approx_screen.shape[0], 2)
        cor_top_left = r_approx[np.asarray(distance_P2Ps((0, 0), r_approx)).argmin()]
        cor_bottom_left = r_approx[np.asarray(distance_P2Ps((w, 0), r_approx)).argmin()]
        cor_top_right = r_approx[np.asarray(distance_P2Ps((0, h), r_approx)).argmin()]
        cor_bottom_right = r_approx[np.asarray(distance_P2Ps((w, h), r_approx)).argmin()]

        box = np.array([cor_top_left, cor_top_right, cor_bottom_right, cor_bottom_left])
        cv.drawContours(img, [box], -1, (255, 0, 0), 3)
    """cv.line(img, cor_top_right, cor_top_left, (255, 0, 0), 3)
    cv.line(img, cor_top_left, cor_bottom_left, (255, 0, 0), 3)
    cv.line(img, cor_bottom_left, cor_bottom_right, (255, 0, 0), 3)
    cv.line(img, cor_top_right, cor_bottom_right, (255, 0, 0), 3)"""

    return img


"""    cv.namedWindow('img', cv.WINDOW_NORMAL)
    cv.imshow('img', img)
    cv.waitKey(0)"""


def use_SIFT(screen_shot, icon):
    return ld.get_location(screen_shot, icon)


"""src = './res/kc1_test_0.jpg'
tar = './res/KC1.jpg'

use_canny(src)
# for i in range(5):
# src = './res/kc_test_' + str(i) + '.jpg'

img = cv.imread(src)

location = use_SIFT(src, tar)
cv.drawContours(img, [location], -1, (0, 255, 0), 3)
cv.namedWindow('img', cv.WINDOW_NORMAL)
cv.imshow('img', img)
cv.waitKey()
"""
