import glob
import pickle
from typing import Tuple

import numpy as np
import cv2

import constants
import video_helper


class Camera:
    def __init__(self, resize_images_by_factor=3):
        self.intrinsic_mat, self.distortion, self.rotation_vector, self.translation_vector = None, None, None, None
        self._resize_images_by_factor = resize_images_by_factor
        self._camera_pickle_name = 'camera_calibration.pkl'

    def undistort(self, img: np.ndarray):
        img = cv2.resize(img,
                         (img.shape[1] // self._resize_images_by_factor, img.shape[0] // self._resize_images_by_factor))
        h, w = img.shape[:2]
        new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(self.intrinsic_mat, self.distortion, (w, h), 1, (w, h))
        dst = cv2.undistort(img, self.intrinsic_mat, self.distortion, None, new_camera_mtx)
        cv2.imshow('img', dst)
        cv2.waitKey(0)

    def calibrate_helper(self, board_size: Tuple[int, int], is_video: bool, cube_mm_size: float):
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, cube_mm_size, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        obj_points = []  # 3d point in real world space
        img_points = []  # 2d points in image plane.

        if is_video:
            videos = glob.glob('camera_calibration/*.mp4')
            images = video_helper.generate_frames(videos[0])
        else:
            images = [cv2.imread(img) for img in glob.glob('camera_calibration/*.jpg')]

        # counter = 0
        for img in images:
            img = cv2.resize(img, (
                img.shape[1] // self._resize_images_by_factor, img.shape[0] // self._resize_images_by_factor))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # cv2.imwrite(f'camera_calibration_images/{counter}.jpg', gray)
            # counter += 1

            # Find the chess board corners
            success, corners = cv2.findChessboardCorners(gray, (board_size[0], board_size[1]), None)

            # If found, add object points, image points (after refining them)
            if success:
                obj_points.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                img_points.append(corners)

                # Draw and display the corners
                cv2.drawChessboardCorners(img, (board_size[0], board_size[1]), corners2, success)
                cv2.imshow('img', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cv2.destroyAllWindows()

        # u = obj_points[1][1] - obj_points[0][1]
        # v = obj_points[2][1] - obj_points[0][1]
        # normal = np.cross(u, v)
        # a, b, c = normal
        #
        # d = np.dot(normal, obj_points[1])
        #
        # print('The equation is {0}x + {1}y + {2}z = {3}'.format(a, b, c, d))

        _, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
        return mtx, dist, rvecs[2], tvecs[2]

    def calibrate(self, board_size: Tuple[int, int], is_video: bool, cube_mm_size: float):
        def start_calibration():
            return self.calibrate_helper(board_size, is_video, cube_mm_size)

        def set_calibration_parameters(params, to_pickle=False):
            self.intrinsic_mat, self.distortion, self.rotation_vector, self.translation_vector = params
            if to_pickle:
                with open(self._camera_pickle_name, 'wb') as f2:
                    pickle.dump(params, f2)

        if not constants.CAMERA_CALIBRATE_PICKLE:
            set_calibration_parameters(start_calibration(), to_pickle=True)
        else:
            try:
                with open(self._camera_pickle_name, 'rb') as f1:
                    print("Found camera_matrix.pkl")
                    set_calibration_parameters(pickle.load(f1))
            except (OSError, IOError):
                set_calibration_parameters(start_calibration(), to_pickle=True)
