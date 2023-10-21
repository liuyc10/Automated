import sys
from functools import partial

from PyQt5 import QtWidgets, QtCore

from ui import kcui, main_window
from default_setting import DefaultSetting


def on_slider_value_change(v):
    print(v)


if __name__ == '__main__':
    # QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = QtWidgets.QMainWindow()
    ui = main_window.Ui_MainWindow()
    ui.setupUi(mainWindow)

    # ui.horizontalSlider_fa.setValue(setting.fa)
    # ui.horizontalSlider_fa.valueChanged.connect(partial(slider_change, ui.horizontalSlider_fa, ui.label_fa_v))

    # ui.checkBox_Debug.clicked.connect(clicked)
    mainWindow.show()
    sys.exit(app.exec_())
