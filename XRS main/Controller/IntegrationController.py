from StdSuites.Standard_Suite import _3c_

__author__ = 'Clemens Prescher'
import sys
import os

from PyQt4 import QtGui, QtCore
from Views.IntegrationView import IntegrationView
from Data.ImgData import ImgData
from Data.MaskData import MaskData
from Data.CalibrationData import CalibrationData
from Data.SpectrumData import SpectrumData
import pyqtgraph as pg
import time
## Switch to using white background and black foreground
pg.setConfigOption('useOpenGL', False)
pg.setConfigOption('leftButtonPan', False)
pg.setConfigOption('background', 'k')
pg.setConfigOption('foreground', 'w')
pg.setConfigOption('antialias', True)
import pyFAI.units
import numpy as np


class IntegrationController(object):
    def __init__(self):
        self.view = IntegrationView()
        self.img_data = ImgData()
        self.mask_data = MaskData()
        self.calibration_data = CalibrationData(self.img_data)
        self.spectrum_data = SpectrumData()
        self.initialize()
        self.create_sub_controller()
        self.img_data.notify()

        self.view.setWindowState(self.view.windowState() & ~QtCore.Qt.WindowMinimized | QtCore.Qt.WindowActive)
        self.view.activateWindow()
        self.view.raise_()

    def initialize(self):
        self.img_data.set_calibration_file('ExampleData/LaB6_p49_001.poni')
        self.img_data.load('ExampleData/Mg2SiO4_ambient_001.tif')
        self.mask_data.set_dimension(self.img_data.get_img_data().shape)
        self.mask_data.set_mask(np.loadtxt('ExampleData/test.mask'))
        self.calibration_data.load('ExampleData/LaB6_p49_40keV_006.poni')

    def create_sub_controller(self):
        self.spectrum_controller = XrsIntegrationSpectrumController(self.view, self.img_data, self.mask_data,
                                                                    self.calibration_data, self.spectrum_data)
        self.file_controller = XrsIntegrationFileController(self.view, self.img_data, self.mask_data)

class XrsIntegrationSpectrumController(object):
    def __init__(self, view, img_data, mask_data, calibration_data, spectrum_data):
        self.view = view
        self.img_data = img_data
        self.mask_data = mask_data
        self.calibration_data = calibration_data
        self.spectrum_data = spectrum_data

        self.create_subscriptions()
        self.spectrum_working_dir = 'ExampleData/spectra'
        self.set_status()

        self.create_signals()

    def create_subscriptions(self):
        self.img_data.subscribe(self.image_changed)
        self.spectrum_data.subscribe(self.plot_spectra)

    def set_status(self):
        self.autosave = True
        self.unit = pyFAI.units.TTH_DEG

    def create_signals(self):
        self.connect_click_function(self.view.spec_autocreate_cb, self.autocreate_cb_changed)
        self.connect_click_function(self.view.spec_load_btn, self.load)
        self.connect_click_function(self.view.spec_previous_btn, self.load_previous)
        self.connect_click_function(self.view.spec_next_btn, self.load_next)
        self.connect_click_function(self.view.spec_browse_by_name_rb, self.set_iteration_mode_number)
        self.connect_click_function(self.view.spec_browse_by_time_rb, self.set_iteration_mode_time)

    def connect_click_function(self, emitter, function):
        self.view.connect(emitter, QtCore.SIGNAL('clicked()'), function)

    def image_changed(self):
        if self.autosave:
            filename = os.path.join(self.spectrum_working_dir,
                                    os.path.basename(self.img_data.filename).split('.')[:-1][0] + '.xy')
            self.view.spec_next_btn.setEnabled(True)
            self.view.spec_previous_btn.setEnabled(True)
            self.view.spec_filename_lbl.setText(os.path.basename(filename))
            self.view.spec_directory_txt.setText(os.path.dirname(filename))
        else:
            self.view.spec_next_btn.setEnabled(False)
            self.view.spec_previous_btn.setEnabled(False)
            filename = None
            self.view.spec_filename_lbl.setText('No File saved or selected')

        tth, I = self.calibration_data.integrate_1d(filename=filename)
        spectrum_filename = os.path.join(self.spectrum_working_dir,
                                         os.path.basename(self.img_data.filename).split('.')[:-1][0] + '.xy')
        self.spectrum_data.set_spectrum(tth, I, spectrum_filename)


    def plot_spectra(self):
        x, y = self.spectrum_data.spectrum.data
        self.view.spectrum_view.plot_data(x, y)


    def load(self, filename=None):
        if filename is None:
            filename = str(QtGui.QFileDialog.getOpenFileName(self.view, caption="Load Spectrum",
                                                             directory=self.spectrum_working_dir))
        if filename is not '':
            self.spectrum_working_dir = os.path.dirname(filename)
            self.view.spec_directory_txt.setText(self.spectrum_working_dir)
            self.view.spec_filename_lbl.setText(os.path.basename(filename))
            self.spectrum_data.load_spectrum(filename)
            self.view.spec_next_btn.setEnabled(True)
            self.view.spec_previous_btn.setEnabled(True)

    def load_previous(self):
        self.spectrum_data.load_previous()
        self.view.spec_filename_lbl.setText(os.path.basename(self.spectrum_data.spectrum_filename))

    def load_next(self):
        self.spectrum_data.load_next()
        self.view.spec_filename_lbl.setText(os.path.basename(self.spectrum_data.spectrum_filename))

    def autocreate_cb_changed(self):
        self.autosave = self.view.spec_autocreate_cb.isChecked()


    def set_iteration_mode_number(self):
        self.spectrum_data.file_iteration_mode = 'number'

    def set_iteration_mode_time(self):
        self.spectrum_data.file_iteration_mode = 'time'


class XrsIntegrationFileController(object):
    def __init__(self, view, img_data, mask_data):
        self.view= view
        self.img_data = img_data
        self.mask_data = mask_data
        self._working_dir = ''
        self._reset_img_levels = False
        self.view.show()
        self.initialize()
        self.img_data.subscribe(self.update_img)
        self.create_signals()

    def initialize(self):
        self.update_img(True)
        self.plot_mask()
        self.view.img_view.img_view_box.autoRange()

    def plot_img(self, reset_img_levels=None):
        if reset_img_levels is None:
            reset_img_levels=self._reset_img_levels
        self.view.img_view.plot_image(self.img_data.get_img_data(), reset_img_levels)
        if reset_img_levels:
            self.view.img_view.auto_range()

    def plot_mask(self):
        if self.view.mask_show_cb.isChecked():
            self.view.img_view.plot_mask(self.mask_data.get_img())
        else:
            self.view.img_view.plot_mask(np.zeros(self.mask_data.get_img().shape))

    def change_mask_colormap(self):
        if self.view.mask_transparent_cb.isChecked():
            self.view.img_view.set_color([255,0,0,100])
        else:
            self.view.img_view.set_color([255,0,0,255])
        self.plot_mask()

    def change_img_levels_mode(self):
        self.view.img_view.img_histogram_LUT.percentageLevel = self.view.img_levels_percentage_rb.isChecked()
        self.view.img_view.img_histogram_LUT.old_hist_x_range = self.view.img_view.img_histogram_LUT.hist_x_range
        self.view.img_view.img_histogram_LUT.first_image = False

    def create_signals(self):
        self.connect_click_function(self.view.next_img_btn, self.load_next_img)
        self.connect_click_function(self.view.prev_img_btn, self.load_previous_img)
        self.connect_click_function(self.view.load_img_btn, self.load_file_btn_click)
        self.connect_click_function(self.view.img_browse_by_name_rb, self.set_iteration_mode_number)
        self.connect_click_function(self.view.img_browse_by_time_rb, self.set_iteration_mode_time)
        self.connect_click_function(self.view.mask_show_cb, self.plot_mask)
        self.connect_click_function(self.view.mask_transparent_cb, self.change_mask_colormap)
        self.connect_click_function(self.view.img_levels_absolute_rb, self.change_img_levels_mode)
        self.connect_click_function(self.view.img_levels_percentage_rb, self.change_img_levels_mode)

    def connect_click_function(self, emitter, function):
        self.view.connect(emitter, QtCore.SIGNAL('clicked()'), function)

    def load_file_btn_click(self, filename = None):
        if filename is None:
            filename = str(QtGui.QFileDialog.getOpenFileName(self.view, caption="Load image data",
                                                             directory=self._working_dir))

        if filename is not '':
            self._working_dir = os.path.dirname(filename)
            self.img_data.load(filename)
            self.plot_img()

    def load_next_img(self):
        self.img_data.load_next()

    def load_previous_img(self):
        self.img_data.load_previous_file()

    def update_img(self, reset_img_levels=False):
        self.plot_img(reset_img_levels)
        self.view.img_filename_lbl.setText(os.path.basename(self.img_data.filename))

    def set_iteration_mode_number(self):
        self.img_data.file_iteration_mode = 'number'

    def set_iteration_mode_time(self):
        self.img_data.file_iteration_mode = 'time'


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    controller = IntegrationController()
    app.exec_()