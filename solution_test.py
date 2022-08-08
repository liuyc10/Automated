import cv2 as cv
import numpy as np
from paddleocr import PaddleOCR

DEBUG = True


def get_texts(ocr, screen_shot):
    global DEBUG

    result = ocr.ocr(screen_shot, cls=False)
    if DEBUG:
        for line in result:
            print(line)
    return result


img = cv.imread('./res/SOLUTION_1080.jpg')
ocr = PaddleOCR(use_angle_cls=False, lang="en")
# blur = cv.GaussianBlur(img, (3, 3), 0)
# cv.imshow('blur', blur)
# cv.waitKey()

texts = get_texts(ocr, img)

"""for i in range(10):

    blur = cv.GaussianBlur(img, (2*i+1, 2*i+1), 0)
    texts = get_texts(ocr, blur)
    # cv.imshow('blur', blur)
    for t in texts:
        if len(t[1][0]) == 38 and '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ' in t[1][0]:
            print(t[1][0][-2:])
            break"""
