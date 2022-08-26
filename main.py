from typing import Tuple

import numpy as np
import cv2 as cv
import glob
import pickle

USE_PICKLE = True


def undistort(mtx: np.ndarray, dist: np.ndarray, img: np.ndarray):
    img = cv.resize(img, (img.shape[1] // 3, img.shape[0] // 3))
    h, w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    dst = cv.undistort(img, mtx, dist, None, newcameramtx)

    cv.imshow('img', dst)
    cv.waitKey(0)


def get_frame_from_video(video_name: str, frame_number: int):
    for i, frame in enumerate(generate_frames(video_name)):
        if i == frame_number:
            return frame


def generate_frames(video_name: str, skip_frames: int = 4):
    cap = cv.VideoCapture(video_name)
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if i % skip_frames == 0:
            yield frame
        i += 1
    cap.release()


def calibrate(board_size: Tuple[int, int], is_video: bool, cube_mm_size: float = 30, resize_img: bool = True):
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, cube_mm_size, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    obj_points = []  # 3d point in real world space
    img_points = []  # 2d points in image plane.

    if is_video:
        videos = glob.glob('./calibration/*.mp4')
        images = generate_frames(videos[0])
    else:
        images = [cv.imread(img) for img in glob.glob('./calibration/*.jpg')]

    for img in images:
        if resize_img:
            img = cv.resize(img, (img.shape[1] // 3, img.shape[0] // 3))
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        success, corners = cv.findChessboardCorners(gray, (board_size[0], board_size[1]), None)

        # If found, add object points, image points (after refining them)
        if success:
            obj_points.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            img_points.append(corners)

            # Draw and display the corners
            cv.drawChessboardCorners(img, (board_size[0], board_size[1]), corners2, success)
            cv.imshow('img', img)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    cv.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
    return ret, mtx, dist, rvecs, tvecs


def start_calibration():
    print("Couldn't find camera_matrix.pkl")
    ret, mtx, dist, rvecs, tvecs = calibrate(board_size=(7, 7), cube_mm_size=20, is_video=True, resize_img=True)
    with open('camera_matrix.pkl', 'wb') as f:
        pickle.dump((ret, mtx, dist, rvecs, tvecs), f)
    return ret, mtx, dist, rvecs, tvecs


def main():
    if not USE_PICKLE:
        ret, mtx, dist, rvecs, tvecs = start_calibration()
    else:
        try:
            with open('camera_matrix.pkl', 'rb') as f:
                print("Found camera_matrix.pkl")
                ret, mtx, dist, rvecs, tvecs = pickle.load(f)
        except (OSError, IOError) as e:
            ret, mtx, dist, rvecs, tvecs = start_calibration()
    undistort(mtx, dist, get_frame_from_video(glob.glob('./calibration/*.mp4')[0], frame_number=50))


if __name__ == '__main__':
    main()
