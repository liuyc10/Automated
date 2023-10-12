import datetime

import cv2 as cv
import numpy as np

from calculation_tools import cal_area

DEBUG = True
minimum = 0
maximum = 1
part_limits = {'pan': (-168, 168),
               'tilt': (-30, 90),
               'zoom': (0, 16384)}

VideoCapturePropertiesName = {0: 'CAP_PROP_POS_MSEC',
                              1: 'CAP_PROP_POS_FRAMES',
                              2: 'CAP_PROP_POS_AVI_RATIO',
                              3: 'CAP_PROP_FRAME_WIDTH',
                              4: 'CAP_PROP_FRAME_HEIGHT',
                              5: 'CAP_PROP_FPS',
                              6: 'CAP_PROP_FOURCC',
                              7: 'CAP_PROP_FRAME_COUNT',
                              8: 'CAP_PROP_FORMAT',
                              9: 'CAP_PROP_MODE',
                              10: 'CAP_PROP_BRIGHTNESS',
                              11: 'CAP_PROP_CONTRAST',
                              12: 'CAP_PROP_SATURATION',
                              13: 'CAP_PROP_HUE',
                              14: 'CAP_PROP_GAIN',
                              15: 'CAP_PROP_EXPOSURE',
                              16: 'CAP_PROP_CONVERT_RGB',
                              17: 'CAP_PROP_WHITE_BALANCE_BLUE_U',
                              18: 'CAP_PROP_RECTIFICATION',
                              19: 'CAP_PROP_MONOCHROME',
                              20: 'CAP_PROP_SHARPNESS',
                              21: 'CAP_PROP_AUTO_EXPOSURE',
                              22: 'CAP_PROP_GAMMA',
                              23: 'CAP_PROP_TEMPERATURE',
                              24: 'CAP_PROP_TRIGGER',
                              25: 'CAP_PROP_TRIGGER_DELAY',
                              26: 'CAP_PROP_WHITE_BALANCE_RED_V',
                              27: 'CAP_PROP_ZOOM',
                              28: 'CAP_PROP_FOCUS',
                              29: 'CAP_PROP_GUID',
                              30: 'CAP_PROP_ISO_SPEED',
                              32: 'CAP_PROP_BACKLIGHT',
                              33: 'CAP_PROP_PAN',
                              34: 'CAP_PROP_TILT',
                              35: 'CAP_PROP_ROLL',
                              36: 'CAP_PROP_IRIS',
                              37: 'CAP_PROP_SETTINGS',
                              38: 'CAP_PROP_BUFFERSIZE',
                              39: 'CAP_PROP_AUTOFOCUS',
                              40: 'CAP_PROP_SAR_NUM',
                              41: 'CAP_PROP_SAR_DEN',
                              42: 'CAP_PROP_BACKEND',
                              43: 'CAP_PROP_CHANNEL',
                              44: 'CAP_PROP_AUTO_WB',
                              45: 'CAP_PROP_WB_TEMPERATUR',
                              46: 'CAP_PROP_CODEC_PIXEL_FORMAT',
                              47: 'CAP_PROP_BITRATE',
                              48: 'CAP_PROP_ORIENTATION_MET',
                              49: 'CAP_PROP_ORIENTATION_AUT',
                              50: 'CAP_PROP_HW_ACCELERATIO',
                              51: 'CAP_PROP_HW_DEVICE',
                              52: 'CAP_PROP_HW_ACCELERATION_USE_OPENC',
                              53: 'CAP_PROP_OPEN_TIMEOUT_MSE',
                              54: 'CAP_PROP_READ_TIMEOUT_MSE',
                              55: 'CAP_PROP_STREAM_OPEN_TIME_USEC',
                              56: 'CAP_PROP_VIDEO_TOTAL_CHANNELS',
                              57: 'CAP_PROP_VIDEO_STREAM',
                              58: 'CAP_PROP_AUDIO_STREAM',
                              59: 'CAP_PROP_AUDIO_POS',
                              60: 'CAP_PROP_AUDIO_SHIFT_NSEC',
                              61: 'CAP_PROP_AUDIO_DATA_DEPTH',
                              62: 'CAP_PROP_AUDIO_SAMPLES_PER_SECOND',
                              63: 'CAP_PROP_AUDIO_BASE_INDEX',
                              64: 'CAP_PROP_AUDIO_TOTAL_CHANNELS',
                              65: 'CAP_PROP_AUDIO_TOTAL_STREAMS',
                              66: 'CAP_PROP_AUDIO_SYNCHRONIZE',
                              67: 'CAP_PROP_LRF_HAS_KEY_FRAME',
                              68: 'CAP_PROP_CODEC_EXTRADATA_INDEX',
                              69: 'CAP_PROP_FRAME_TYPE',
                              70: 'CAP_PROP_N_THREADS'
                              }

VideoCapturePropertiesName_abbr = {'width': cv.CAP_PROP_FRAME_WIDTH,
                                   'height': cv.CAP_PROP_FRAME_HEIGHT,
                                   'fps': cv.CAP_PROP_FPS,
                                   'brightness': cv.CAP_PROP_BRIGHTNESS,
                                   'contrast': cv.CAP_PROP_CONTRAST,
                                   'saturation': cv.CAP_PROP_SATURATION,
                                   'hue': cv.CAP_PROP_HUE,
                                   'gain': cv.CAP_PROP_GAIN,
                                   'exposure': cv.CAP_PROP_EXPOSURE,
                                   'auto_exposure': cv.CAP_PROP_AUTO_EXPOSURE,
                                   'gamma': cv.CAP_PROP_GAMMA,
                                   'temperature': cv.CAP_PROP_TEMPERATURE,
                                   'zoom': cv.CAP_PROP_ZOOM,
                                   'focus': cv.CAP_PROP_FOCUS,
                                   'pan': cv.CAP_PROP_PAN,
                                   'tilt': cv.CAP_PROP_TILT,
                                   'setting': cv.CAP_PROP_SETTINGS,
                                   'auto_focus': cv.CAP_PROP_AUTOFOCUS,
                                   'auto_wb': cv.CAP_PROP_AUTO_WB,

                                   6: 'CAP_PROP_FOURCC',
                                   7: 'CAP_PROP_FRAME_COUNT',
                                   8: 'CAP_PROP_FORMAT',
                                   9: 'CAP_PROP_MODE',
                                   16: 'CAP_PROP_CONVERT_RGB',
                                   17: 'CAP_PROP_WHITE_BALANCE_BLUE_U',
                                   18: 'CAP_PROP_RECTIFICATION',
                                   19: 'CAP_PROP_MONOCHROME',
                                   20: 'CAP_PROP_SHARPNESS',
                                   24: 'CAP_PROP_TRIGGER',
                                   25: 'CAP_PROP_TRIGGER_DELAY',
                                   26: 'CAP_PROP_WHITE_BALANCE_RED_V',
                                   27: 'CAP_PROP_ZOOM',
                                   28: 'CAP_PROP_FOCUS',
                                   29: 'CAP_PROP_GUID',
                                   30: 'CAP_PROP_ISO_SPEED',
                                   32: 'CAP_PROP_BACKLIGHT',

                                   35: 'CAP_PROP_ROLL',
                                   36: 'CAP_PROP_IRIS',
                                   38: 'CAP_PROP_BUFFERSIZE',
                                   40: 'CAP_PROP_SAR_NUM',
                                   41: 'CAP_PROP_SAR_DEN',
                                   42: 'CAP_PROP_BACKEND',
                                   43: 'CAP_PROP_CHANNEL',

                                   45: 'CAP_PROP_WB_TEMPERATUR',
                                   46: 'CAP_PROP_CODEC_PIXEL_FORMAT',
                                   47: 'CAP_PROP_BITRATE',
                                   48: 'CAP_PROP_ORIENTATION_MET',
                                   49: 'CAP_PROP_ORIENTATION_AUT',
                                   50: 'CAP_PROP_HW_ACCELERATIO',
                                   51: 'CAP_PROP_HW_DEVICE',
                                   52: 'CAP_PROP_HW_ACCELERATION_USE_OPENC',
                                   53: 'CAP_PROP_OPEN_TIMEOUT_MSE',
                                   54: 'CAP_PROP_READ_TIMEOUT_MSE',
                                   55: 'CAP_PROP_STREAM_OPEN_TIME_USEC',
                                   56: 'CAP_PROP_VIDEO_TOTAL_CHANNELS',
                                   57: 'CAP_PROP_VIDEO_STREAM',
                                   58: 'CAP_PROP_AUDIO_STREAM',
                                   59: 'CAP_PROP_AUDIO_POS',
                                   60: 'CAP_PROP_AUDIO_SHIFT_NSEC',
                                   61: 'CAP_PROP_AUDIO_DATA_DEPTH',
                                   62: 'CAP_PROP_AUDIO_SAMPLES_PER_SECOND',
                                   63: 'CAP_PROP_AUDIO_BASE_INDEX',
                                   64: 'CAP_PROP_AUDIO_TOTAL_CHANNELS',
                                   65: 'CAP_PROP_AUDIO_TOTAL_STREAMS',
                                   66: 'CAP_PROP_AUDIO_SYNCHRONIZE',
                                   67: 'CAP_PROP_LRF_HAS_KEY_FRAME',
                                   68: 'CAP_PROP_CODEC_EXTRADATA_INDEX',
                                   69: 'CAP_PROP_FRAME_TYPE',
                                   70: 'CAP_PROP_N_THREADS''ex'

                                   # 'gamma': 22,
                                   }


class Camera(object):

    def __init__(self, camera_no=0, **kwargs):

        self.v = cv.VideoCapture(camera_no, cv.CAP_DSHOW)
        self.default_properties = self.supported_properties()
        self.properties = self.default_properties.copy()

        if kwargs:
            self.set_properties(kwargs)

        self.h = self.height()
        self.w = self.width()
        self.area = self.h * self.w

    def height(self):
        return self.properties[VideoCapturePropertiesName_abbr['height']]

    def width(self):
        return self.properties[VideoCapturePropertiesName_abbr['width']]

    def isOpened(self):
        return self.v.isOpened()

    def read(self, skip_df=True):
        if skip_df:
            self.v.read()       # skip duplicate frame
        return self.v.read()

    def release(self):
        self.v.release()

    def skip(self, sec):
        fps = 20
        skip_count = fps * sec
        start = datetime.datetime.now()
        f = None
        while skip_count > 0:
            if self.v.isOpened():
                r, f = self.v.read(False)
                skip_count -= 1
            else:
                print('camera disconnected')
                break
        end = datetime.datetime.now()
        print('skipped {} sec'.format((end - start).seconds))
        return f

    def set_properties(self, kwargs):
        for key, value in kwargs.items():
            p = VideoCapturePropertiesName_abbr[key]
            if p in self.properties:
                self.v.set(p, value)
                self.properties[p] = value
                print('{} has been set to {}'.format(p, value))

            else:
                print("This camera doesn't support {} control".format(key))

    def update_pos(self, part_name):
        value = None
        part = VideoCapturePropertiesName_abbr[part_name]
        if part in self.properties:
            self.properties[part] = self.v.get(part)
            value = self.properties[part]
        else:
            print("This camera doesn't support {} control".format(part_name))
        return value

    def movement_control(self, part_name, steps):
        part = VideoCapturePropertiesName_abbr[part_name]
        limits = part_limits[part_name]
        current = self.properties[part]
        if current <= limits[minimum] and steps < 0:
            print("Minimum {} value reached, won't change".format(part_name))
            return 'M-'

        if current >= limits[maximum] and steps > 0:
            print("Maximum {} value reached, won't change".format(part_name))
            return 'M+'

        if steps == 0:
            new = 0
        else:
            new = current + steps
            if new >= limits[maximum]:
                new = limits[maximum]
            if new <= limits[minimum]:
                new = limits[minimum]
        self.v.set(part, new)
        self.properties[part] = new
        return new

    def reset_pos(self, part_name, value=0):
        new_value = None
        if part_name is not None:
            new_value = self.movement_control(part_name, value)
        return new_value

    def pan_update(self):
        return self.update_pos('pan')

    def pan_control(self, steps=1):
        return self.movement_control('pan', steps)

    def pan_reset(self, steps=0):
        return self.reset_pos('pan', steps)

    def tilt_update(self):
        return self.update_pos('tilt')

    def tilt_control(self, steps=1):
        return self.movement_control('tilt', steps)

    def tilt_reset(self, steps=0):
        return self.reset_pos('tilt', steps)

    def zoom_update(self):
        return self.update_pos('zoom')

    def zoom_control(self, zoom_step=1):
        return self.movement_control('zoom', zoom_step)

    def zoom_reset(self, steps=0):
        return self.reset_pos('zoom', steps)

    def all_pos_reset(self):
        self.pan_reset()
        self.tilt_reset()
        self.zoom_reset()

    def targeting_coordinate(self, target, current) -> bool:
        pan_move = 0
        tilt_move = 0

        offset_x, offset_y = np.asarray(current) - target

        if abs(offset_x) > 200:
            pan_move = (offset_x - (offset_x / abs(offset_x)) * 100) // 100

        if abs(offset_y) > 200:
            tilt_move = -(offset_y - (offset_y / abs(offset_y)) * 100) // 100

        if pan_move or tilt_move:
            self.pan_control(pan_move)
            self.tilt_control(tilt_move)
            self.skip(1)
            return True
        else:
            return False

    def precise_targeting(self, current_box) -> bool:

        pan_move = 0
        tilt_move = 0

        tl, tr, br, bl = current_box
        tl_x, tl_y = tl
        tr_x, tr_y = tr
        br_x, br_y = br
        bl_x, bl_y = bl
        t_y = min(tl_y, tr_y)
        b_y = max(bl_y, br_y)
        offset_y = t_y - (self.h - b_y)
        l_x = min(tl_x, bl_x)
        r_x = max(tr_x, br_x)
        offset_x = l_x - (self.w - r_x)

        if abs(offset_x) > 200:
            pan_move = (offset_x - (offset_x / abs(offset_x)) * 100) // 100
        # elif offset_x < -200:
        #     pan_move = -1

        if abs(offset_y) > 200:
            tilt_move = -(offset_y - (offset_y / abs(offset_y)) * 100) // 100
        # elif offset_y < -200:
        #     tilt_move = 1
        print('x: {}, y:{}'.format(offset_x, offset_y))

        if pan_move or tilt_move:
            self.pan_control(pan_move)
            self.tilt_control(tilt_move)
            self.skip(1)
            return True
        else:
            return False

    def zoom(self, current_box) -> bool:

        tl, tr, br, bl = current_box
        tl_x, tl_y = tl
        tr_x, tr_y = tr
        br_x, br_y = br
        bl_x, bl_y = bl

        if min(tl_x, tl_y, self.w - tr_x, tr_y, bl_x, self.h - bl_y, self.w - br_x, self.h - br_y) > 200:
            self.zoom_control(300)
            self.skip(1)
            return True
        elif min(tl_x, tl_y, self.w - tr_x, tr_y, bl_x, self.h - bl_y, self.w - br_x, self.h - br_y) < 100:
            self.zoom_control(50)
            self.skip(1)
            return True
        else:
            return False

    def supported_properties(self) -> dict:
        pro_dict = dict()
        for i in range(70):
            pro_value = self.v.get(i)
            if pro_value == -1:
                continue
            else:
                pro_dict[i] = pro_value
                if DEBUG:
                    print("No.={} {} = {}".format(i, VideoCapturePropertiesName[i], pro_value))
        return pro_dict

    def properties_changed(self):
        for key, value in self.default_properties.items():
            new_value = self.v.get(key)
            if new_value != value:
                print("No.={} {} = {}".format(key, VideoCapturePropertiesName[key], new_value))

    def print_properties_diff(self):
        for i in range(70):
            new_value = self.v.get(i)
            if i not in self.default_properties:
                if new_value != -1:
                    print("No.={} {} = {}".format(i, VideoCapturePropertiesName[i], new_value))
                else:
                    continue
            elif new_value != self.default_properties[i]:
                print("No.={} {} = {}".format(i, VideoCapturePropertiesName[i], new_value))
