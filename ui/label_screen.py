from PyQt5 import QtGui
from PyQt5.QtWidgets import QLabel


class LabelScreen(QLabel):

    def __init__(self, parent=None):
        super(LabelScreen, self).__init__(parent)
        self.show_flag = True

    def show_img(self, img):
        if self.show_flag:
            image = QtGui.QImage(img[:], img.shape[1], img.shape[0], img.shape[1]*3, QtGui.QImage.Format_RGB888)
            out = QtGui.QPixmap(image)
            self.setPixmap(out)

    def stop(self):
        self.show_flag = False

    def resume(self):
        self.show_flag = True
