import datetime

import cv2 as cv
import numpy as np
from scipy.spatial.distance import pdist
import csv

import calculation_tools
import tools

DEBUG = False
prev = None
set_done = False
enable_setting = True


def show_img(img, name='img'):
    cv.namedWindow(name, cv.WINDOW_NORMAL)
    cv.imshow(name, img)
    cv.waitKey(10)


def searching(image):
    global prev

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    r, threshold_screen = cv.threshold(gray.copy(), 200, 255, cv.THRESH_BINARY)
    if DEBUG:
        show_img(threshold_screen)
    blur_screen = cv.GaussianBlur(threshold_screen, (11, 11), 0)
    canny_screen = cv.Canny(blur_screen, threshold1=200, threshold2=250)
    contours_screen, _ = cv.findContours(canny_screen, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    cv.drawContours(image, contours_screen, -1, (0, 0, 255), 10)
    return contours_screen


def ex_change(x):
    pass


def start_test(x, s):
    global set_done, writer
    set_done = not set_done
    writer.writeheader()


def createTrackbar(name, win, min_value, max_value, default_value):
    cv.createTrackbar(name, win, int(default_value), int(max_value - min_value), ex_change)
    cv.setTrackbarMin(name, win, int(min_value))
    cv.setTrackbarMax(name, win, int(max_value))
    cv.setTrackbarPos(name, win, int(default_value))


def videoSetting(cap, exposure=None, brightness=None, gamma=None, width=1920, height=1080, auto_focus=None,
                 setting=False):
    cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
    if exposure:
        cap.set(cv.CAP_PROP_EXPOSURE, exposure)
    if brightness:
        cap.set(cv.CAP_PROP_BRIGHTNESS, brightness)
    if gamma:
        cap.set(cv.CAP_PROP_GAMMA, gamma)
    if auto_focus:
        cap.set(cv.CAP_PROP_AUTOFOCUS, auto_focus)
    if setting:
        cap.set(cv.CAP_PROP_SETTINGS, 1)

    # return cap


def support_properties(cap):
    pro_dict = dict()
    for i in range(47):
        pro_value = cap.get(i)
        if pro_value == -1:
            continue
        else:
            pro_dict[i] = pro_value
            if DEBUG:
                print("No.={} {} = {}".format(i, tools.VideoCapturePropertiesName[i], pro_value))
    return pro_dict


def set_property(cap, prop, old_value, new_value):
    if old_value != new_value:
        old_value = new_value
        cap.set(prop, new_value)


def reset_status(status):
    new_status = dict()
    for k in status.keys():
        new_status[k] = 0
    return status


def add_TimeStamp(image, ts, coordinates):
    cv.putText(image, ts, (100, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    if len(lights_coordinate) != 0:
        for number, coordinate in coordinates.items():
            # cv.circle(frame, coordinate.astype(int), 25, (0, 0, 255), 3)
            cv.putText(image, str(number), (coordinate + [-50, 10]).astype(int), cv.FONT_HERSHEY_SIMPLEX, 1,
                       (255, 0, 255), 2)


if __name__ == '__main__':
    # source = 'live'
    # source = './res/WIN_20220810_16_46_23_Pro.mp4'

    default_height = 1080
    default_width = 1920

    lights_coordinate = dict()
    lights_status = dict()
    coordinate_tolerance = 0
    properties = dict()
    light_count = 0

    v = cv.VideoCapture(0, cv.CAP_DSHOW)
    videoSetting(v, height=default_height, width=default_width, setting=True)

    default_properties = support_properties(v)

    if enable_setting:
        properties[cv.CAP_PROP_EXPOSURE] = [default_properties[cv.CAP_PROP_EXPOSURE], -15]
        properties[cv.CAP_PROP_FOCUS] = [default_properties[cv.CAP_PROP_FOCUS], 40]
        properties[cv.CAP_PROP_AUTOFOCUS] = [default_properties[cv.CAP_PROP_AUTOFOCUS], 40]
        properties[cv.CAP_PROP_BRIGHTNESS] = [default_properties[cv.CAP_PROP_BRIGHTNESS], 100]
        # properties[cv.CAP_PROP_GAMMA] = [default_properties_value[cv.CAP_PROP_GAMMA], 300]

        cv.namedWindow('frame', cv.WINDOW_NORMAL)
        cv.namedWindow('settings', cv.WINDOW_AUTOSIZE)
        createTrackbar('ex', 'settings', -35, 0, properties[cv.CAP_PROP_EXPOSURE][0])  # 15
        # createTrackbar('gamma', 'settings', -1000, 1000, properties[cv.CAP_PROP_GAMMA][0])  # 22
        createTrackbar('brightness', 'settings', -1000, 1000, properties[cv.CAP_PROP_BRIGHTNESS][0])  # 10
        createTrackbar('focus', 'settings', 0, 200, properties[cv.CAP_PROP_FOCUS][0])  # 25
        createTrackbar('auto_focus', 'settings', -5, 5, properties[cv.CAP_PROP_AUTOFOCUS][0])  # 39
        cv.createButton('start', start_test)

    t = datetime.datetime.now()
    timestamp = '{}{}{}_{}_{}_{}'.format(t.year, t.month, t.day, t.hour, t.minute, t.second)
    f = open('./data/log_{}.csv'.format(timestamp), mode='w', newline='')
    writer = None
    prev_frame = None
    skip_frame_no = 0
    properties_changed = True
    try:
        while v.isOpened():

            if properties_changed:
                for p, l in properties.items():
                    set_property(v, p, l[0], l[1])

            if DEBUG:
                for key, value in default_properties.items():
                    n_value = v.get(key)
                    if value != n_value:
                        print("No.{} {} = {}".format(key, tools.VideoCapturePropertiesName[key], n_value))
                        default_properties[key] = n_value

            rat, frame = v.read()
            if frame is None:
                break

            save_img = False
            t = datetime.datetime.now()
            timestamp = '{}{}{}_{}{}{}{}'.format(t.year, t.month, t.day,
                                                 t.hour, t.minute, t.second,
                                                 t.microsecond if len(str(t.microsecond)) == 6 else '0' + str(
                                                     t.microsecond))

            if skip_frame_no > 0:
                skip_frame_no -= 1
                add_TimeStamp(frame, timestamp, lights_coordinate)
                cv.imshow('frame', frame)
                continue

            circles = searching(frame)

            centers = np.asarray(
                [calculation_tools.cal_central_point(circle.reshape(circle.shape[0], 2)) for circle in circles])

            if not set_done:
                lights_coordinate = dict()
                lights_status = dict()
                num = 0
                for center in centers:
                    lights_coordinate[num] = center
                    lights_status[num] = 1
                    num += 1
                light_count = len(centers)
                if len(centers) < 2:
                    coordinate_tolerance = 40
                else:
                    distances = pdist(centers, 'euclidean')
                    coordinate_tolerance = min(distances) / 2
                heads = ['Time Stamp']
                heads.extend(lights_status.keys())
                writer = csv.DictWriter(f, heads)
            else:
                lights_status_new = dict()
                # if len(centers) > len(lights_status):
                #    continue

                for center in centers:
                    distances = calculation_tools.distance_P2Ps(center, lights_coordinate.values())
                    min_index = np.argmin(distances)
                    if distances[min_index] < coordinate_tolerance:
                        lights_status_new[min_index] = 1

                diff = set(lights_status) - set(lights_status_new)

                if diff:
                    for key in diff:
                        lights_status_new[key] = 0

                for key in lights_status.keys():
                    if lights_status[key] != lights_status_new[key]:
                        lights_status[key] = lights_status_new[key]
                        print('{} Light No. {} status has been changed to {}'.format(
                            datetime.datetime.now(),
                            key,
                            "OFF" if lights_status[key] == 0 else 'ON'))
                        save_img = True

            if save_img:
                add_TimeStamp(frame, timestamp, lights_coordinate)

                cv.imwrite('./data/{}.jpg'.format(timestamp), frame)
                cv.imwrite('./data/{}_p.jpg'.format(prev_frame[1]), prev_frame[0])

                record = {'Time Stamp': timestamp}
                for key, value in lights_status.items():
                    record[key] = value
                writer.writerow(record)
                skip_frame = 15

            prev_frame = (frame, timestamp)
            cv.imshow('frame', frame)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

            if enable_setting:
                properties_changed = False
                for Trackbar_name, key in tools.VideoCapturePropertiesName_abbr.items():
                    if properties[key][1] != cv.getTrackbarPos(Trackbar_name, 'settings'):
                        properties[key][1] = cv.getTrackbarPos(Trackbar_name, 'settings')
                        properties_changed = True

            '''
            properties[cv.CAP_PROP_EXPOSURE][1] = cv.getTrackbarPos('ex', 'frame')
            properties[cv.CAP_PROP_FOCUS] = cv.getTrackbarPos('ex', 'frame')
            properties[cv.CAP_PROP_AUTOFOCUS] = cv.getTrackbarPos('ex', 'frame')
            properties[cv.CAP_PROP_BRIGHTNESS] = cv.getTrackbarPos('ex', 'frame')
            properties[cv.CAP_PROP_GAMMA] = cv.getTrackbarPos('ex', 'frame')

            e_n = cv.getTrackbarPos('ex', 'frame')
            f_n = cv.getTrackbarPos('focus', 'frame')
                        '''

    finally:
        f.close()
        v.release()
