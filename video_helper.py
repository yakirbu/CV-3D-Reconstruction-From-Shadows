import glob
from typing import Generator, Optional

import cv2
import numpy as np


def generate_frames(video_name: str, skip_frames: int = 4) -> Generator[np.ndarray, None, None]:
    cap = cv2.VideoCapture(video_name)
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if i % skip_frames == 0:
            print(i)
            yield frame
        i += 1
    cap.release()


def get_frame_from_video(video_name: str, frame_number: int) -> Optional[np.ndarray]:
    for i, frame in enumerate(generate_frames(video_name)):
        if i == frame_number:
            return frame

def get_camera_calibration_images():
    return [cv2.imread(img) for img in glob.glob('./camera_calibration_images/*.jpg')]
