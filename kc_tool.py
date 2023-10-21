import os
import sys
from functools import partial

from PyQt5 import QtWidgets

import kc
from default_setting import DefaultSetting
from ui import kcui


class KeystoneCorrectionTool(object):

    def __init__(self):
        self.kc = kc.KeystoneCorrection()

        app = QtWidgets.QApplication(sys.argv)
        mainWindow = QtWidgets.QMainWindow()
        self.ui = kcui.KcUi()
        self.ui.setup_ui(mainWindow)
        self.initial_ui()

        self.kc.set_screen(self.ui.label_main, self.ui.label_preprocess, self.ui.label_edge, self.ui.label_cross)
        mainWindow.show()
        sys.exit(app.exec_())

    def initial_ui(self):

        self.ui.radioButton_source_live.clicked.connect(self.on_radio_button_clicked)
        self.ui.radioButton_source_file.clicked.connect(self.on_radio_button_clicked)

        # connect slider and label
        for slider, label in self.ui.slider_label_dict.items():
            slider.valueChanged.connect(partial(self.on_slider_value_change, slider, label))
            slider.sliderPressed.connect(partial(self.on_slider_pressed, slider, label))
            slider.sliderReleased.connect(partial(self.on_slider_released, slider, label))

        dataset = DefaultSetting()

        for slider, data in zip(self.ui.slider_label_dict.keys(), dataset.data_set()):
            slider.setValue(data)

        self.ui.pushButton_file_select.clicked.connect(self.on_push_button_file_select_clicked)
        self.ui.pushButton_start.clicked.connect(self.on_push_button_start_clicked)
        self.ui.pushButton_stop.clicked.connect(self.on_push_button_stop_clicked)
        self.ui.pushButton_pause_resume.clicked.connect(self.on_push_button_pause_resume_clicked)

    def gathering_settings(self):
        debug = self.ui.checkBox_DEBUG.isChecked()
        video = None
        camera_no = 0
        w = 0
        h = 0

        if self.ui.radioButton_source_live.isChecked():
            camera_no = self.ui.spinBox_CAM_NO.value()
            if self.ui.comboBox_resolution.currentIndex() == 0:
                w = 1920
                h = 1080
            else:
                w = 3840
                h = 2160
        else:
            video = self.ui.lineEdit_file_path.text()

        properties_dict = dict()
        properties_dict['frame'] = self.ui.horizontalSlider_frame.value()
        properties_dict['gauss'] = self.ui.horizontalSlider_gauss.value()
        properties_dict['th1sr'] = self.ui.horizontalSlider_th1sr.value()
        properties_dict['th2sr'] = self.ui.horizontalSlider_th2sr.value()
        properties_dict['th1cr'] = self.ui.horizontalSlider_th1cr.value()
        properties_dict['th2cr'] = self.ui.horizontalSlider_th2cr.value()
        properties_dict['th1ca'] = self.ui.horizontalSlider_th1ca.value()
        properties_dict['th2ca'] = self.ui.horizontalSlider_th2ca.value()
        properties_dict['scale'] = self.ui.horizontalSlider_scale.value()

        self.kc.video_setting(video=video, camera_no=camera_no, width=w, height=h, setting=debug)
        self.kc.property_setting(properties_dict)

    def show_input_dialog(self):
        working_path = os.getcwd()
        fileinfo = QtWidgets.QFileDialog.getOpenFileName(self.ui.central_widget, 'select file', working_path,
                                                         'video file(*.mp4)')
        # fileinfo = QtWidgets.QFileDialog.getOpenFileName(self.ui.central_widget, 'select file', working_path,
        # 'video file(*.mp4);video f(*.mkv)')
        return fileinfo[0]

    def on_radio_button_clicked(self):
        if self.ui.radioButton_source_live.isChecked():
            self.ui.comboBox_resolution.setEnabled(True)
            self.ui.spinBox_CAM_NO.setEnabled(True)
            self.ui.lineEdit_file_path.setEnabled(False)
            self.ui.pushButton_file_select.setEnabled(False)
        else:
            self.ui.comboBox_resolution.setEnabled(False)
            self.ui.spinBox_CAM_NO.setEnabled(False)
            self.ui.lineEdit_file_path.setEnabled(True)
            self.ui.pushButton_file_select.setEnabled(True)

    def on_slider_value_change(self, slider, label, slider_v):
        label.setText(str(slider_v))
        self.kc.property_set_single(slider.objectName(), slider_v)

    def on_slider_released(self, slider, label):
        slider.sliderMoved.disconnect()
        slider.valueChanged.connect(partial(self.on_slider_value_change, slider, label))
        self.on_slider_value_change(slider, label, slider.value())

    def on_slider_moved(self, label, slider_v):
        label.setText(str(slider_v))

    def on_slider_pressed(self, slider, label):
        slider.valueChanged.disconnect()
        slider.sliderMoved.connect(partial(self.on_slider_moved, label))

    def on_push_button_file_select_clicked(self):
        self.ui.lineEdit_file_path.setText(self.show_input_dialog())

    def on_push_button_start_clicked(self):
        self.ui.pushButton_pause_resume.setEnabled(True)
        self.ui.pushButton_start.setEnabled(False)
        self.ui.pushButton_stop.setEnabled(False)
        self.gathering_settings()
        self.kc.start()

    def on_push_button_stop_clicked(self):
        self.ui.pushButton_pause_resume.setEnabled(False)
        self.ui.pushButton_start.setEnabled(True)
        self.ui.pushButton_stop.setEnabled(False)

    def on_push_button_pause_resume_clicked(self):
        if self.ui.pushButton_pause_resume.text() == 'Pause':
            self.ui.label_main.stop()
            self.ui.label_preprocess.stop()
            self.ui.label_edge.stop()
            self.ui.label_cross.stop()
            self.ui.pushButton_pause_resume.setText('Resume')
        else:
            self.ui.label_main.resume()
            self.ui.label_preprocess.resume()
            self.ui.label_edge.resume()
            self.ui.label_cross.resume()
            self.ui.pushButton_pause_resume.setText('Pause')


if __name__ == '__main__':
    a = KeystoneCorrectionTool()
