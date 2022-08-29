import numpy as np

from camera import Camera
import video_helper
from light_source import LightSource


def main():
    camera = Camera()
    camera.calibrate(board_size=(7, 7), is_video=True, cube_mm_size=20)

    print(camera.intrinsic_mat)
    light_source = LightSource(camera=camera)
    light_source.calibrate(pencil_len_mm=139.7)

    print(light_source.light_position)

if __name__ == '__main__':
    main()
