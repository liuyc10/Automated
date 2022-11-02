from time import sleep

import cv2 as cv
import numpy as np
from coordinatequeue import CoordinateQueue

import keystone_correction as kc
from tools import DataWriter


def get_video(src):
    if src == 'live':
        v = cv.VideoCapture(0, cv.CAP_DSHOW)
        v.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
        v.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
    else:
        v = cv.VideoCapture(src)
    # cv.VideoCapture(0, cv.CAP_DSHOW)
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


def mark_corners(src, start_frame=0, frame_count=3, writer=None):
    boxes = []
    sleep(5)
    corners_queue = CoordinateQueue(frame_count)
    cross_queue = CoordinateQueue(frame_count)
    frame_no = 0
    while src.isOpened():
        ret, frame = src.read()

        if frame is None:
            src.release()
            if writer is not None:
                writer.write_dataset(boxes)
                writer.save()
            break

        if frame_no < start_frame:
            frame_no += 1
            continue

        corners = kc.get_edges(frame.copy())
        cross = kc.get_cross(frame.copy(), kc.make_mask(corners, 5))
        if corners is not None and cross is not None:
            corners_queue.push(corners)
            cross_queue.push(cross)
            corners_avg = corners_queue.average()
            cross_avg = cross_queue.average()
            cv.drawContours(frame, [corners_avg.astype(int)], -1, (0, 255, 0), 1)
            cv.drawContours(frame, [cross_avg.astype(int)], -1, (0, 0, 255), 1)
            extend_cross = kc.get_extend_rec(corners_avg, cross_avg)
            cv.drawContours(frame, [extend_cross.astype(int)], -1, (255, 255, 0), 1)

            cv.imshow('capture', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            src.release()
            if writer is not None:
                writer.write_dataset(boxes)
                writer.save()
            break


def mark_corners_org(src, start_frame=0, writer=None):
    boxes = []
    sleep(5)
    frame_list = []
    frame_no = 0
    while src.isOpened():
        ret, frame = src.read()

        if frame is None:
            src.release()
            if writer is not None:
                writer.write_dataset(boxes)
                writer.save()
            break

        if frame_no < start_frame:
            frame_no += 1
            continue

        # new_frame = kc.use_canny(frame)
        # cv.imshow('capture', new_frame)

        # mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        # mask[540:, :] = 255
        # new_frame, corners = kc.get_edges(frames, mask=mask)

        new_frame, corners = kc.get_edges(frame.copy())
        cv.imshow('capture', new_frame)
        _, cross = kc.get_cross(frame.copy(), kc.make_mask(corners, 5))
        input_str = cv.waitKey(1)
        if input_str == ord('y'):
            path = './data/' + str(frame_no) + '.jpg'
            boxes.append(corners)
            cv.imwrite(path, frame)
            frame_list.append(path)
            frame_no += 1
        elif input_str == ord('n'):
            frame_no += 1
            continue
        elif input_str == ord('p'):
            cv.waitKey()
        elif input_str == ord('q'):
            src.release()
            if writer is not None:
                writer.write_dataset(zip(frame_list, boxes))
                writer.save()
            break

        if cv.waitKey(1) & 0xFF == ord('q'):
            src.release()
            if writer is not None:
                writer.write_dataset(boxes)
                writer.save()
            break


def mark_focus(src, writer=None):
    boxes = []
    sleep(5)

    while src.isOpened():
        ret, frame = src.read()

        if frame is None:
            src.release()
            if writer is not None:
                writer.write_dataset(boxes)
                writer.save()
            break

        new_frame, corners = kc.get_cross(frame.copy())
        cv.imshow('capture', new_frame)


if __name__ == '__main__':
    source = 'live'
    # source = './res/WIN_20220810_16_46_23_Pro.mp4'
    # source = './res/WIN_20221018_16_15_15_Pro.mp4'
    # source = r'D:\development\WIN_20221024_11_09_10_Pro.mp4'
    # source = './res/WIN_20221031_11_21_21_Pro.mp4'
    # source = './res/WIN_20221101_11_25_17_Pro.mp4'
    cap = get_video(source)
    data_writer = None
    # data_writer = DataWriter('./data/data.xlsx')
    try:
        # for i in range(5, 19, 2):
        # mark_corners_org(cap, writer=data_writer)
        # cap = get_video(source)
        # except Exception as ex:
        #    print(ex)
        #    cap.release()

        mark_corners(cap, frame_count=5)

    finally:
        cap.release()
        if data_writer is not None:
            data_writer.save_and_close()
