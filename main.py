from camera import Camera
import video_helper

def main():
    camera = Camera()
    camera.calibrate(board_size=(7, 7), is_video=True, cube_mm_size=20)
    camera.undistort(video_helper.get_frame_from_video('camera_calibration/calib1.mp4', 5))

if __name__ == '__main__':
    main()
