from time import sleep

import cv2 as cv
import numpy as np

from camera_control import Camera
from coordinatequeue import CoordinateQueue

import keystone_correction as kc
from tools import DataWriter

writer = False
use_avg = False
tolerance_calculation = True
max_number = 0

global_frame = None


def get_video(src='live'):
    if src == 'live':
        v = cv.VideoCapture(0, cv.CAP_DSHOW)
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


def videoSetting(src, exposure=None, brightness=None, gamma=None, width=None, height=None, auto_focus=None,
                 setting=False):
    src.set(cv.CAP_PROP_FRAME_WIDTH, width)
    src.set(cv.CAP_PROP_FRAME_HEIGHT, height)
    if exposure:
        src.set(cv.CAP_PROP_EXPOSURE, exposure)
    if brightness:
        src.set(cv.CAP_PROP_BRIGHTNESS, brightness)
    if gamma:
        src.set(cv.CAP_PROP_GAMMA, gamma)
    if auto_focus:
        src.set(cv.CAP_PROP_AUTOFOCUS, auto_focus)
    if setting:
        src.set(cv.CAP_PROP_SETTINGS, 1)


def targeting_corner(src, shape, index, current_coordinate):
    h, w, d = shape
    coordinate = current_coordinate
    x_offset = coordinate[0] - w / 2
    y_offset = coordinate[1] - h / 2

    if abs(x_offset) > 500:
        src.pan_control(x_offset // 47)
    if abs(y_offset) > 500:
        src.tilt_control(-y_offset // 47)
    new_frame = src.skip(3)
    coordinate = kc.get_edges(new_frame.copy())[index]
    x_offset = coordinate[0] - w / 2
    y_offset = coordinate[1] - h / 2

    cv.circle(new_frame, coordinate, 10, (255, 255, 255), 5)
    cv.imshow('capture', new_frame)
    cv.waitKey(10)

    if abs(x_offset) > 50 or abs(y_offset) > 50:
        if abs(x_offset) > 50:
            src.pan_control(x_offset / (abs(x_offset) * 2))
        if abs(y_offset) > 50:
            src.tilt_control(-y_offset / (abs(y_offset) * 2))
        new_frame = src.skip(0.1)
        coordinate = kc.get_edges(new_frame.copy())[index]
        cv.circle(new_frame, coordinate, 10, (255, 255, 255), 5)
        cv.imshow('capture', new_frame)
        cv.waitKey(10)
        targeting_corner(src, new_frame.shape, index, coordinate)
    else:
        return


def focus_on_corner_x(src, frame, index, current_coordinate):
    h, w, d = frame.shape
    coordinate = current_coordinate
    x_offset = coordinate[0] - w / 2

    print(x_offset)
    if abs(x_offset) > 500:
        src.pan_control(x_offset // 47)
        sleep(2)
        ret, new_frame = src.read()
        coordinate = kc.get_edges(new_frame.copy())[index]
        cv.circle(new_frame, coordinate, 10, (255, 255, 255), 5)
        cv.imshow('capture', new_frame)
        cv.waitKey()
        x_offset = coordinate[0] - w / 2

    if abs(x_offset) > 50:
        src.pan_control(x_offset / (abs(x_offset) * 2))
        ret, new_frame = src.read()

        coordinate = kc.get_edges(new_frame.copy())[index]

        cv.circle(new_frame, coordinate, 10, (255, 255, 255), 5)
        cv.imshow('capture', new_frame)
        cv.waitKey(10)
        focus_on_corner_x(src, new_frame, index, coordinate)

    else:
        return


def focus_on_corner_y(src, frame, index, current_coordinate):
    h, w, d = frame.shape
    y_offset = current_coordinate[1] - h / 2
    print(y_offset)

    if abs(y_offset) > 500:
        src.tilt_control(-y_offset // 47)
        sleep(2)
        ret, new_frame = src.read()
        cv.imshow('capture', new_frame)
        corners = kc.get_edges(new_frame)
        y_offset = corners[index][1] - h / 2
    cv.waitKey()
    if abs(y_offset) > 50:
        src.tilt_control(-y_offset / (abs(y_offset) * 2))

        ret, new_frame = src.read()
        cv.imshow('capture', frame)
        corners = kc.get_edges(new_frame)
        focus_on_corner_y(src, new_frame, index, corners[index])

    else:
        return


def mark_corners(src, start_frame=0, frame_count=3, writer=None):
    boxes = []
    sleep(5)
    frame_path_list = []
    corners_queue = CoordinateQueue(frame_count)
    cross_queue = CoordinateQueue(frame_count)
    frame_no = 0
    while src.isOpened():
        ret, frame_org = src.read()
        frame = frame_org.copy()
        if frame is None:
            src.release()
            if writer is not None:
                writer.write_dataset(zip(frame_path_list, boxes))
                writer.save()
            break

        if frame_no < start_frame:
            frame_no += 1
            continue
        frame_no += 1
        corners = kc.get_edges(frame.copy())
        if corners is not None:
            corners_avg = corners
            if tolerance_calculation:
                cross = kc.get_cross(frame.copy(), kc.make_mask(corners, tuple(frame.shape[:2]), 10))
                if cross is not None:
                    if use_avg:
                        corners_queue.push(corners)
                        cross_queue.push(cross)
                        corners_avg = corners_queue.average()
                        cross_avg = cross_queue.average()
                    else:
                        cross_avg = cross
                    cv.drawContours(frame, [corners_avg.astype(int)], -1, (0, 255, 0), 1)
                    cv.drawContours(frame, [cross_avg.astype(int)], -1, (0, 0, 255), 1)
                    extend_cross, lines = kc.get_extend_rec(corners_avg, cross_avg)
                    cv.drawContours(frame, [extend_cross.astype(int)], -1, (255, 255, 0), 1)
                    cross_length = kc.cal_length(cross_avg)
                    kc.draw_side_length(frame, cross_avg, cross_length)
                    org_corners_length = kc.cal_length(corners_avg)
                    kc.draw_side_length(frame, corners, org_corners_length)
                    cal_corners_length = kc.length_calibration(corners_avg, cross_avg)
                    # tb_tolerance, lr_tolerance = kc.cal_tolerance(corners_length)
                    kc.draw_side_length(frame, corners, cal_corners_length, -150)
                    kc.draw_tolerance(frame, corners, lines)
                    cv.imshow('capture', frame)

            if writer is not None:
                path = './data/' + str(frame_no) + '.jpg'
                boxes.append(corners)
                cv.imwrite(path, frame_org)
                # print(path + '      ' + str(result))
                frame_path_list.append(path)

        if cv.waitKey(1) & 0xFF == ord('q'):
            src.release()
            if writer is not None:
                writer.write_dataset(zip(frame_path_list, boxes))
                writer.save()
            break

        if max_number == 0:
            continue

        if frame_no > max_number:
            src.release()
            if writer is not None:
                writer.write_dataset(zip(frame_path_list, boxes))
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
            result = cv.imwrite(path, frame)
            if result:
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

        corners = kc.get_edges(frame.copy())

        cv.imshow('capture', frame)
        cv.waitKey(10)
        if corners is not None:
            targeting_corner(src, frame.shape, 0, corners[0])
            cv.waitKey()
            src.zoom_control(16384)
            frame = src.skip(5)

            cv.imshow('capture', frame)
            cv.waitKey(1000)
            src.all_pos_reset()
            frame = src.skip(3)
            cv.imshow('capture', frame)
            cv.waitKey(1000)

        if cv.waitKey(1) & 0xFF == ord('q'):
            src.release()
            break


def new_m():
    image = cv.imread('./res/cross_3.jpg')
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blur_screen = cv.GaussianBlur(gray, (5, 5), 0)
    canny_screen = cv.Canny(blur_screen, threshold1=50, threshold2=100)
    contours_screen, _ = cv.findContours(canny_screen, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    i = 0
    temp = contours_screen[-1]
    for contour in contours_screen:
        m = cv.moments(contour)
        hm = cv.HuMoments(m)

        score1 = cv.matchShapes(contour, temp, cv.CONTOURS_MATCH_I1, 0)
        score2 = cv.matchShapes(contour, temp, cv.CONTOURS_MATCH_I2, 0)
        score3 = cv.matchShapes(contour, temp, cv.CONTOURS_MATCH_I3, 0)

        print('{}. score 1= {}, score2= {}, score3= {}'.format(i, score1, score2, score3))
        # print('{}. '.format(i), end='')
        # print(hm)
        cx = m['m10'] / m['m00']
        cy = m['m01'] / m['m00']
        cv.putText(image, str(i), np.array([cx, cy]).astype(int), cv.FONT_HERSHEY_SIMPLEX, 1,
                   (255, 0, 255), 2)
        i += 1
    cv.namedWindow('show', cv.WINDOW_NORMAL)
    cv.drawContours(image, contours_screen, -1, (0, 255, 0), 1)
    cv.imshow('show', image)
    cv.waitKey()
    print('done')


def frame_read(src):
    global global_frame

    while src.isOpened():
        rat, globe_frame = src.read()


def t(src):
    while src.isOpened():
        r, frame = src.read()
        if frame is None:
            src.release()
            break
        corners = kc.get_edges(frame.copy())
        if corners is None:
            continue
        cv.drawContours(frame, [corners], -1, (0, 255, 0), 1)
        cv.imshow('capture', frame)
        cv.waitKey(10)


if __name__ == '__main__':
    source = 'live'
    # source = './res/WIN_20220810_16_46_23_Pro.mp4'
    # source = './res/WIN_20221018_16_15_15_Pro.mp4'
    # source = r'D:\development\WIN_20221024_11_09_10_Pro.mp4'
    # source = './res/WIN_20221031_11_21_21_Pro.mp4'
    # source = './res/WIN_20221101_11_25_17_Pro.mp4'

    # new_m()
    cv.namedWindow('capture', cv.WINDOW_NORMAL)
    cap = Camera(width=3840, height=2160, auto_focus=2, setting=1)
    cap.all_pos_reset()
    if writer:
        data_writer = DataWriter('./data/data.xlsx')
    else:
        data_writer = None
    try:
        # for i in range(5, 19, 2):
        # mark_corners_org(cap, writer=data_writer)
        # cap = get_video(source)
        # except Exception as ex:
        #    print(ex)
        #    cap.release()
        mark_focus(cap)
        # mark_corners(cap, frame_count=5, writer=data_writer)
        # t(cap)

    finally:
        cap.release()
        if data_writer is not None:
            data_writer.save_and_close()
