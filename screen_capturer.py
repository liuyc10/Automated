from time import sleep

import cv2 as cv
import keystone_correction as kc


cap = cv.VideoCapture(0, cv.CAP_DSHOW)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
frames = []

while cap.isOpened():
    ret, frame = cap.read()

    if frame is None:
        cap.release()
        break
    frames.append(frame)
    # new_frame = kc.use_canny(frame)
    # cv.imshow('capture', new_frame)

    if len(frames) == 3:
        new_frame = kc.use_canny(frames)
        frames = []
        cv.imshow('capture', new_frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
