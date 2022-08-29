import constants
from camera import Camera
from light_source import LightSource


def main():
    camera = Camera()
    camera.calibrate(board_size=constants.CALIB_BOARD_SIZE,
                     is_video=True,
                     cube_mm_size=constants.CALIB_CUBE_MM_SIZE)

    print(camera.intrinsic_mat)
    light_source = LightSource(camera=camera)
    light_source.calibrate(pencil_len_mm=constants.PENCIL_LENGTH_MM)

    print(light_source.light_position)

if __name__ == '__main__':
    main()
