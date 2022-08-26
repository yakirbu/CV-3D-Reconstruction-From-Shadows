from typing import Tuple

import numpy as np
import cv2 as cv
import glob
import pickle

USE_PICKLE = True


def undistort(mtx: np.ndarray, dist: np.ndarray):
    img = cv.imread('old2/20220826_132012.jpg')
    img = cv.resize(img, (img.shape[1] // 3, img.shape[0] // 3))
    h, w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    dst = cv.undistort(img, mtx, dist, None, newcameramtx)

    cv.imshow('img', dst)
    cv.waitKey(0)


def calibrate(board_size: Tuple[int, int], cube_mm_size: float = 30, resize_img: bool = True):
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, cube_mm_size, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    obj_points = []  # 3d point in real world space
    img_points = []  # 2d points in image plane.
    images = glob.glob('./calibration/*.jpg')

    for image_name in images:
        img = cv.imread(image_name)
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
            cv.waitKey(0)

    cv.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
    return ret, mtx, dist, rvecs, tvecs


def start_calibration():
    print("Couldn't find camera_matrix.pkl")
    ret, mtx, dist, rvecs, tvecs = calibrate(board_size=(7, 7), cube_mm_size=20, resize_img=True)
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
    undistort(mtx, dist)


if __name__ == '__main__':
    main()
