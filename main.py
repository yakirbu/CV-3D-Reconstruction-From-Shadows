import glob
import pickle

import numpy as np

import constants
import video_helper
from camera import Camera
from light_source import LightSource
from shadow_edge_detection import ShadowEdgeDetection

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
from PyQt6 import QtWidgets
import sys


# app = QtWidgets.QApplication(sys.argv)
# view = gl.GLViewWidget()
# view.show()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.view = gl.GLViewWidget()  # pg.PlotWidget()
        self.setCentralWidget(self.view)

        xaxis = gl.GLAxisItem(size=QtGui.QVector3D(100, 100, 100))
        yaxis = gl.GLAxisItem()
        zaxis = gl.GLAxisItem()
        xgrid = gl.GLGridItem()
        ygrid = gl.GLGridItem()
        zgrid = gl.GLGridItem()
        self.view.addItem(xaxis)
        self.view.addItem(yaxis)
        self.view.addItem(zaxis)
        self.view.addItem(xgrid)
        self.view.addItem(ygrid)
        self.view.addItem(zgrid)

        # set window size to 1080x720
        self.view.setFixedSize(1080, 720)


def main():
    # create opengl window
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()

    print("##############################")
    print("Make sure this information is correct:")
    print(f"1. Board size: {constants.CALIB_BOARD_SIZE}")
    print(f"2. Board cube size (mm): {constants.CALIB_CUBE_MM_SIZE}")
    print(f"3. Pencil Length: {constants.PENCIL_LENGTH_MM}")
    print("##############################\n\n")

    camera = Camera(opengl_app=app, opengl_window=window)
    camera.calibrate(is_video=True, cube_mm_size=constants.CALIB_CUBE_MM_SIZE)
    camera.calculate_camera_center()

    light_source = LightSource(camera=camera)
    light_source.calibrate(pencil_len_mm=constants.PENCIL_LENGTH_MM)

    print(f"camera-center: {camera.cam_center}")
    print(f"light-position: {light_source.light_position}")

    if constants.LOAD_CALIBRATED_DATA:
        # with open('final_calibration.pkl', 'wb') as f:
        #     pickle.dump((np.array([0, 600, 289]).reshape(-1, 1), np.array([50, 720, 1280]).reshape(-1, 1)), f)
        with open('final_calibration.pkl', 'rb') as f:
            camera.cam_center, light_source.light_position = pickle.load(f)

    shadow_edge_detection = ShadowEdgeDetection(camera=camera, light=light_source)
    shadow_edge_detection.detect()


if __name__ == '__main__':
    main()
