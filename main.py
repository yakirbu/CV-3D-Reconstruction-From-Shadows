import glob
import pickle

import numpy as np

import constants
from camera import Camera
from light_source import LightSource
from shadow_edge_detection import ShadowEdgeDetection


def main():
    print("##############################")
    print("Make sure this information is correct:")
    print(f"1. Board size: {constants.CALIB_BOARD_SIZE}")
    print(f"2. Board cube size (mm): {constants.CALIB_CUBE_MM_SIZE}")
    print(f"3. Pencil Length: {constants.PENCIL_LENGTH_MM}")
    print("##############################\n\n")

    camera = Camera()
    camera.calibrate(is_video=True, cube_mm_size=constants.CALIB_CUBE_MM_SIZE)
    camera.calculate_camera_center()

    light_source = LightSource(camera=camera)
    light_source.calibrate(pencil_len_mm=constants.PENCIL_LENGTH_MM)

    print(f"camera-center: {camera.cam_center}")
    print(f"light-position: {light_source.light_position}")

    shadow_edge_detection = ShadowEdgeDetection(camera=camera, light=light_source)
    shadow_edge_detection.detect()


if __name__ == '__main__':
    main()
