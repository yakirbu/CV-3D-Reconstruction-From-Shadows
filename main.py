from camera import Camera
import video_helper
from light_source import LightSource


def main():
    camera = Camera()
    camera.calibrate(board_size=(7, 7), is_video=True, cube_mm_size=20)
    # camera.undistort(video_helper.get_frame_from_video('camera_calibration/calib1.mp4', 5))
    light_source = LightSource(camera)
    light_source.calibrate()

if __name__ == '__main__':
    main()
