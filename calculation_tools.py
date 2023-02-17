import numpy as np
from cv2 import moments


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
        m = moments(pts)
        if m['m00'] == 0.0:
            return np.median(pts, axis=0)
        cx = m['m10'] / m['m00']
        cy = m['m01'] / m['m00']
    return cx, cy


'''
def cal_central_point(pts):
    if len(pts) < 3:
        return np.uint8((pts[0] + pts[-1]) / 2)
    else:
        area = cal_area(pts)
        xc, yc = 0.0, 0.0
        x0, y0 = pts[-1]
        for x1, y1 in pts:
            xc += ((x1 + x0) * (y1 * x0 - y0 * x1))
            yc += ((y1 + y0) * (y1 * x0 - y0 * x1))
            y0, x0 = y1, x1
        return np.asarray([xc / (6 * area), yc / (6 * area)])
'''


def get_line(pt0, pt1):
    x0, y0 = pt0
    x1, y1 = pt1
    return np.asarray([y1 - y0, x0 - x1, x1 * y0 - x0 * y1])


def get_perpendicular_line(pt, line):
    x, y = pt
    a, b, c = line
    return np.asarray([b, -a, a * y - b * x])


def get_cross_pt(line0, line1):
    a0, b0, c0 = line0
    a1, b1, c1 = line1
    y = (c1 * a0 - c0 * a1) / (b0 * a1 - b1 * a0)
    x = (c1 * b0 - c0 * b1) / (a0 * b1 - a1 * b0)
    return np.asarray([x, y])


def cal_get_perpendicular_line_length(pt, line):
    perpendicular_line = get_perpendicular_line(pt, line)
    cross_pt = get_cross_pt(line, perpendicular_line)
    return distance_P2P(pt, cross_pt)


def get_parallel_line(pt, line):
    x, y = pt
    a, b, c = line
    c = -(a * x + b * y)
    return np.asarray([a, b, c])
