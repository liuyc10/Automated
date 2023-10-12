import datetime
import time


from pathlib import Path
from datetime import datetime


class DataWriter:
    def __init__(self, path):
        self.path = path
        self.app = xw.App(visible=False, add_book=False)
        if not Path(path).exists():
            self.wb = self.app.books.add()
        else:
            self.wb = self.app.books.open(path)
        self.sheet = None
        self.row = 1
        self.col = 1

    def write(self, frame_no, data, sheet_name=None):
        if self.sheet is None:
            if sheet_name:
                self.sheet = self.wb.sheets.add(sheet_name)
            else:
                sheet_name = self.wb.sheets.count
                self.sheet = self.wb.sheets.add(str(sheet_name))

        self.sheet.range(self.row, self.col).value = frame_no
        self.sheet.range(self.row, self.col + 1).value = data.flatten()
        self.row += 1

    def writeDict(self, timestamp, data, sheet_name=None):
        if self.sheet is None:
            if sheet_name:
                self.sheet = self.wb.sheets.add(sheet_name)
            else:
                sheet_name = self.wb.sheets.count
                self.sheet = self.wb.sheets.add(str(sheet_name))

        self.sheet.range(self.row, self.col).value = timestamp
        for key, value in data.items():
            self.sheet.range(self.row, self.col+key+1).value = value
        self.row += 1

    def write_dataset(self, write_dataset):
        print('start writing data......', end=' ')
        row_no = 1
        sheet_name = self.wb.sheets.count
        sheet = self.wb.sheets.add(str(sheet_name))
        for frame_path, data in write_dataset:
            sheet.range(row_no, 1).value = frame_path
            sheet.range(row_no, 2).value = data.flatten()
            row_no += 1
        print('Done')

    def save(self):
        self.wb.save(self.path)

    def save_and_close(self):
        self.wb.save(self.path)
        self.wb.close()
        self.app.quit()


app = None
wb = None
sht = None
cell = None

'''https://github.com/opencv/opencv/blob/master/modules/videoio/include/opencv2/videoio.hpp'''
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

VideoCapturePropertiesName_abbr = {'ex': 15,
                                   'auto_focus': 39,
                                   'focus': 28,
                                   # 'gamma': 22,
                                   'brightness': 10}


def set_offset(box, offset=0, axis=2):  # axis: 0: x axis outside; 1: y axis outside; 2: x and y axis outside

    if axis == 0:
        return box + [[offset, 0],
                      [-offset, 0],
                      [-offset, 0],
                      [offset, 0]]
    elif axis == 1:
        return box + [[0, offset],
                      [0, offset],
                      [0, -offset],
                      [0, -offset]]
    elif axis == 2:
        return box + [[offset, offset],
                      [-offset, offset],
                      [-offset, -offset],
                      [offset, -offset]]
    else:
        return None


def init(path):
    global app, wb, sht, cell
    app = xw.App(visible=False, add_book=False)

    if not Path(path).exists():
        wb = app.books.add()
    else:
        wb = app.books.open(path)
    sheet = wb.sheets.add(str(datetime.datetime.now()))
    cell = sheet.range('A1')


def add_data(data):
    global app, wb, sht, cell
    cell.value = data.flat()
    cell = cell(1, 0)


def write_to_file(data_set, path):
    global app
    if app is None:
        app = xw.App(visible=False, add_book=False)

    working_file = Path(path)

    if working_file.suffix != '.xlsx':
        print('wrong file type!')
        raise NotImplementedError

    if not working_file.exists():
        wb = app.books.add()
        sht = wb.sheets.add(str(datetime.datetime.now()))
    else:
        wb = app.books.open(path)
        sht = wb.sheets.add(str(datetime.datetime.now()))


def time(full=True):
    t = datetime.now()
    if full:
        return '{}{}{}_{}{}{}{}'.format(t.year, t.month, t.day,
                                        t.hour, t.minute, t.second,
                                        t.microsecond if len(str(t.microsecond)) == 6 else '0' + str(
                                            t.microsecond))
    else:
        return '{}{}{}_{}_{}_{}'.format(t.year, t.month, t.day, t.hour, t.minute, t.second)

