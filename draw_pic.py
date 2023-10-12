import cv2 as cv
import numpy as np

step = 1
h = 1080
w = 1920
img = np.zeros((1080, 1920))
start = 0
long = 100

for i in range(1, 10):
    for j in range(3):
        offset = start + i * step
        img[start:offset, :offset] += 1
        img[:offset, start:offset] += 1
        # img[h-offset:, start:offset] += 1
        img[h - offset:h - start, :offset] += 1
        img[:offset, h - offset:h - start] += 1
        start += i * step * 2
        print(start)

cv.imshow('w', img)
cv.waitKey()
