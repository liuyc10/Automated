import datetime
import time

import xlwings as xw
from pathlib import Path


class DataWriter:
    def __init__(self, path):
        self.path = path
        self.app = xw.App(visible=False, add_book=False)
        if not Path(path).exists():
            self.wb = self.app.books.add()
        else:
            self.wb = self.app.books.open(path)
        self.sheet = None
        self.cell = None
        self.row = 1
        self.col = 1

    def write(self, frame_no, data):
        if self.sheet is None:
            sheet_name = self.wb.sheets.count
            self.sheet = self.wb.sheets.add(str(sheet_name))
            self.cell = self.sheet.range('A1')

        self.sheet.range(self.row, self.col).value = frame_no
        self.sheet.range(self.row, self.col + 1).value = data.flatten()
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
