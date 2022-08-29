import numpy as np

from camera import Camera
import video_helper
from light_source import LightSource


def main():
    camera = Camera()
    camera.calibrate(board_size=(7, 7), is_video=True, cube_mm_size=20)

    # camera.intrinsic_mat = np.array([[586.379551545554, 0, 0], [0, 0, -680.217814461694], [2488.70643359556, 48.8943350171507, 1]])
    # camera.rotation_vector = np.array([-0.0991497109220257, 0.183270000643628, -0.225597009557623])
    # camera.translation_vector = np.array([-645.091112119241, 2552.78857431097, 184.030545500584])

    print(camera.intrinsic_mat)
    #camera.undistort(video_helper.get_camera_calibration_images()[-1])
    light_source = LightSource(camera=camera)
    light_source.calibrate(pencil_len_mm=139.7)

if __name__ == '__main__':
    main()
