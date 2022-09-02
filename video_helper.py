import glob
from typing import Generator, Optional

import cv2
import numpy as np
from matplotlib import pyplot as plt

import constants

img_saver_counter = {}


def generate_frames(video_name: str, skip_frames: int = 4, grayscale=False) -> Generator[
    np.ndarray, None, None]:
    cap = cv2.VideoCapture(video_name)
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if i % skip_frames == 0:

            if grayscale:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            yield resize_frame(frame)
        i += 1
    cap.release()


def resize_frame(frame):
    return cv2.resize(frame, (frame.shape[1] // constants.IMG_RESIZE_FACTOR,
                              frame.shape[0] // constants.IMG_RESIZE_FACTOR))


def get_frame_from_video(video_name: str, frame_number: int) -> Optional[np.ndarray]:
    for i, frame in enumerate(generate_frames(video_name)):
        if i == frame_number:
            return frame


def get_images(path):
    return [resize_frame(cv2.imread(img)) for img in glob.glob(f"{path}/*.jpg")]


def get_camera_calibration_images():
    return get_images('./camera_calibration_images')


def get_light_calibration_images():
    return get_images('./light_calibration_images')


def imshow(frame, to_wait):
    cv2.imshow("frame", frame)
    if to_wait:
        cv2.waitKey()
        return False
    else:
        return cv2.waitKey(1) & 0xFF == ord('q')


def save_image(frame, path):
    if path not in img_saver_counter:
        img_saver_counter[path] = 1
    else:
        img_saver_counter[path] += 1
    cv2.imwrite(f"{path}/{img_saver_counter[path]}.jpg", frame)


def show_pixel_selection(camera, frame: np.ndarray, click_callback, title, is_gray=False):
    """
    :param title: title of the window
    :param is_gray: if True, the frame is shown in a gray scale
    :param frame: frame to show
    :param click_callback(x,y), where x,y are the coordinates of the clicked pixel
    """

    def on_click(event):
        if event.dblclick:
            click_callback(event.xdata, event.ydata)
            plt.close()

    # frame = camera.undistort(frame)
    fig = plt.figure()
    fig.suptitle(title, fontsize=14)
    fig.canvas.mpl_connect('button_press_event', on_click)
    if is_gray:
        plt.imshow(frame, cmap='gray')
    else:
        plt.imshow(frame)
    plt.show()
