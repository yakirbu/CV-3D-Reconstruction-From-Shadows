import glob

import constants
import video_helper
from camera import Camera
from light_source import LightSource
from shadow_edge_detection import ShadowEdgeDetection


def main():
    camera = Camera()
    camera.calibrate(board_size=constants.CALIB_BOARD_SIZE,
                     is_video=True,
                     cube_mm_size=constants.CALIB_CUBE_MM_SIZE)
    camera.calculate_camera_center()
    # for p in glob.glob('./camera_calibration_images/*.jpg'):
    #     print(p)
    # camera.undistort(video_helper.get_camera_calibration_images()[-1])
    # camera.undistort(video_helper.get_light_calibration_images()[2])

    print(camera.intrinsic_mat)
    light_source = LightSource(camera=camera)
    light_source.calibrate(pencil_len_mm=constants.PENCIL_LENGTH_MM)

    print(light_source.light_position)

    shadow_edge_detection = ShadowEdgeDetection(camera=camera, light=light_source)
    shadow_edge_detection.detect()


if __name__ == '__main__':
    main()
