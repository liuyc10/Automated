from time import sleep

import cv2 as cv
import numpy as np

import keystone_correction as kc
from tools import DataWriter


def get_video(src):
    v = cv.VideoCapture(src)
    # cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
    # cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
    return v


def video_show(src):
    while src.isOpened():
        ret, frame = src.read()
        if frame is None:
            src.release()
            break
        cv.imshow('capture', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break


def mark(src, frame_count, data_writer):
    frames = []
    boxes = []
    sleep(5)

    while src.isOpened():
        ret, frame = src.read()

        if frame is None:
            src.release()
            data_writer.write_dataset(boxes)
            data_writer.save()
            break
        frames.append(frame)
        # new_frame = kc.use_canny(frame)
        # cv.imshow('capture', new_frame)

        if len(frames) == frame_count:
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            mask[540:, :] = 255
            new_frame, corners = kc.get_edges(frames, mask=mask)
            frames = []
            cv.imshow('capture', new_frame)
            boxes.append(corners)

        if cv.waitKey(1) & 0xFF == ord('q'):
            src.release()
            data_writer.write_dataset(boxes)
            data_writer.save()
            break


if __name__ == '__main__':
    source = './res/WIN_20220810_16_46_23_Pro.mp4'
    cap = get_video(source)
    data_writer = DataWriter('./data/data.xlsx')
    try:
        for i in range(5, 19, 2):
            mark(cap, i, data_writer)
            cap = get_video(source)
    # except Exception as ex:
    #    print(ex)
    #    cap.release()
    finally:
        cap.release()
        data_writer.save_and_close()
