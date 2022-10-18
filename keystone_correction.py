import cv2 as cv
import numpy as np
import location_detemination as ld

DEBUG = True
GAUSS = 5
THRESHOLD1 = 100
THRESHOLD2 = 250
cor_left = [0, 0]
cor_right = [1919, 0]


def distance(src, tar):
    return [(src[0] - t[0]) ** 2 + (src[1] - t[1]) ** 2 for t in tar]


def show_img(img, name='img'):
    cv.namedWindow(name, cv.WINDOW_NORMAL)
    cv.imshow(name, img)
    cv.waitKey(10)


def get_ruler(img):
    red = img.copy()
    red[:, :, 0] = 0
    red[:, :, 1] = 0

    return red


def get_gray(imgs):
    gray = np.zeros((1080, 1920))
    for img in imgs:
        g = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray += g
    gray /= len(imgs)

    return np.unit8(gray)


def get_edges(images, mask=None):
    global GAUSS, THRESHOLD2, THRESHOLD1, DEBUG, cor_right, cor_left
    box = []
    image = images[0]
    gray = []
    for im in images:
        gray.append(cv.cvtColor(im, cv.COLOR_BGR2GRAY))

    r, threshold_screen = cv.threshold(gray[0].copy(), 100, 255, cv.THRESH_TOZERO)
    if DEBUG:
        show_img(threshold_screen)
    blur_screen = cv.GaussianBlur(threshold_screen, (GAUSS, GAUSS), 0)
    canny_screen = cv.Canny(blur_screen, threshold1=THRESHOLD1, threshold2=THRESHOLD2)

    s = np.zeros(image.shape[:2])
    for g in gray:
        revers = cv.bitwise_not(g)
        rr, threshold_temp = cv.threshold(revers, 215, 255, cv.THRESH_BINARY)
        if mask is not None:
            threshold_temp = cv.bitwise_and(threshold_temp, mask).copy()
        s += threshold_temp

    s = s / len(gray)

    threshold_ruler = s.astype(np.uint8)

    if DEBUG:
        show_img(threshold_ruler, 'ruler')
    # rr, threshold_ruler = cv.threshold(threshold_ruler, 127, 255, cv.THRESH_BINARY)
    blur_ruler = cv.GaussianBlur(threshold_ruler, (GAUSS, GAUSS), 0)
    canny_ruler = cv.Canny(blur_ruler, threshold1=100, threshold2=250)

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
    if DEBUG:
        cv.drawContours(image, contours_ruler, -1, (0, 0, 255), 5)

    h, w, d = image.shape

    if len(contours_flat_ruler) != 0:
        approx_ruler = cv.approxPolyDP(np.asarray(contours_flat_ruler), 1080, True)

        r_approx_ruler = approx_ruler.reshape(approx_ruler.shape[0], 2)

        for cor in r_approx_ruler:
            if cor[0] == 0:  # and cor[1] > 700:
                cor_left = cor
            elif cor[0] == 1919:  # and cor[1] > 700:
                cor_right = cor
        cv.line(image, cor_left, cor_right, (0, 255, 0), 5)

    if len(contours_flat_screen) != 0:
        approx_screen = cv.approxPolyDP(np.asarray(contours_flat_screen), 10, True)
        r_approx = approx_screen.reshape(approx_screen.shape[0], 2)
        cor_top_left = r_approx[np.asarray(distance((0, 0), r_approx)).argmin()]
        cor_bottom_left = r_approx[np.asarray(distance((w, 0), r_approx)).argmin()]
        cor_top_right = r_approx[np.asarray(distance((0, h), r_approx)).argmin()]
        cor_bottom_right = r_approx[np.asarray(distance((w, h), r_approx)).argmin()]
        box = np.array([cor_top_left, cor_top_right, cor_bottom_right, cor_bottom_left])
        cv.drawContours(image, [box], -1, (255, 0, 0), 3)

    return image, box


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
        cor_top_left = r_approx[np.asarray(distance((0, 0), r_approx)).argmin()]
        cor_bottom_left = r_approx[np.asarray(distance((w, 0), r_approx)).argmin()]
        cor_top_right = r_approx[np.asarray(distance((0, h), r_approx)).argmin()]
        cor_bottom_right = r_approx[np.asarray(distance((w, h), r_approx)).argmin()]

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
