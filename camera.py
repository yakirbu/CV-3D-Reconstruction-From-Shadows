import glob
import pickle
from typing import Tuple

import numpy as np
import cv2
from matplotlib import pyplot as plt
import pyqtgraph.opengl as gl

import constants
import video_helper


class Camera:
    def __init__(self, opengl_app, opengl_window):
        self.intrinsic_mat, self.distortion, self.rotation_vector, self.translation_vector = None, None, None, None
        self._camera_pickle_name = 'camera_calibration.pkl'
        self.graph = None
        self.cam_center = None
        self.opengl_window = opengl_window
        self.opengl_app = opengl_app

    def undistort(self, img: np.ndarray):
        h, w = img.shape[:2]
        new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(self.intrinsic_mat, self.distortion, (w, h), 1, (w, h))
        dst = cv2.undistort(img, self.intrinsic_mat, self.distortion, None, new_camera_mtx)
        cv2.imshow('img', dst)
        cv2.waitKey(0)

    def calibrate_helper(self, is_video: bool, cube_mm_size: float):
        cbrow = constants.CALIB_BOARD_SIZE[0]
        cbcol = constants.CALIB_BOARD_SIZE[1]

        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, cube_mm_size, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        # objp = np.zeros((constants.CALIB_BOARD_SIZE[0] * constants.CALIB_BOARD_SIZE[1], 3), np.float32)
        # objp[:, :2] = np.mgrid[0:constants.CALIB_BOARD_SIZE[0], 0:constants.CALIB_BOARD_SIZE[1]].T.reshape(-1, 2)
        objp = np.zeros((cbrow * cbcol, 3), np.float32)
        objp[:, :2] = np.mgrid[0:cbcol, 0:cbrow].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        obj_points = []  # 3d point in real world space
        img_points = []  # 2d points in image plane.

        if is_video:
            videos = glob.glob('camera_calibration/*.mp4')
            images = video_helper.generate_frames(videos[0])
        else:
            images = video_helper.get_images('./camera_calibration')

        for img in images:
            video_helper.save_image(img, './camera_calibration_images')

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            success, corners = cv2.findChessboardCorners(gray, (cbcol, cbrow), None)

            # If found, add object points, image points (after refining them)
            if success:
                obj_points.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
                img_points.append(corners)

                # Draw and display the corners
                cv2.drawChessboardCorners(img, (constants.CALIB_BOARD_SIZE[0], constants.CALIB_BOARD_SIZE[1]), corners2, success)
                cv2.imshow('img', img)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break

                # cv2.waitKey()

        cv2.destroyAllWindows()

        # # multiply each element of obj_points by cube_mm_size
        # obj_points = [np.multiply(obj_points[i], cube_mm_size) for i in range(len(obj_points))]
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

        min_err_ind = self.evaluate_calibration(obj_points, img_points, rvecs, tvecs, mtx, dist)

        return mtx, dist, rvecs[min_err_ind], tvecs[min_err_ind]

    def evaluate_calibration(self, obj_points, img_points, rvecs, tvecs, mtx, dist):
        # Print the camera calibration error
        error = 0
        min_err = 1000
        min_err_ind = -1
        for i in range(len(obj_points)):
            imgPoints2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
            curr_error = cv2.norm(img_points[i], imgPoints2, cv2.NORM_L2) / len(imgPoints2)
            error += curr_error
            if curr_error < min_err:
                min_err = curr_error
                min_err_ind = i
            # print(curr_error)
        print("Total calibration error: ", error / len(obj_points))

        # Change min_err_ind to -1 if you want to use the last calibration
        min_err_ind = -1

        self.create_3d_graph()

        if constants.PLOT_CHESSBOARD_POINTS:
            for p in obj_points[min_err_ind]:
                self.add_graph_point(p, 'b')

        return min_err_ind

    def calibrate(self, is_video: bool, cube_mm_size: float):
        def start_calibration():
            return self.calibrate_helper(is_video, cube_mm_size)

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

    def calculate_camera_center(self):
        # Set camera center:
        self.cam_center = -(np.linalg.inv(np.asmatrix(self.get_rotation_matrix())) @ np.asmatrix(self.translation_vector))
        if constants.PLOT_CAMERA_CENTER:
            self.add_graph_point(self.cam_center, color='g')  # plot the camera center on the figure

    def get_rotation_matrix(self):
        # return self.rotation_vector
        return cv2.Rodrigues(self.rotation_vector)[0]

    def image_point_to_3d(self, point_2d):
        # We assume that Z is 0, because our desk is at XY plane
        z_const = 0

        # Transform 2d points to homogeneous coordinates
        point_2d = np.array([[point_2d[0], point_2d[1], 1]], dtype=np.float32).T

        # Get rotation matrix
        rotation_mat = self.get_rotation_matrix()
        rotation_mat_inv = np.linalg.inv(rotation_mat)

        # Get translation vector
        translation_vector, intrinsic_matrix = self.translation_vector.reshape(3, 1), self.intrinsic_mat
        intrinsic_matrix_inv = np.linalg.inv(intrinsic_matrix)

        mat_left_side = rotation_mat_inv @ intrinsic_matrix_inv @ point_2d
        mat_right_side = rotation_mat_inv @ translation_vector

        # Find s:
        s = ((z_const + mat_right_side[2]) / mat_left_side[2])[0]

        # Calculate 3d points:
        P = (s * mat_left_side) - mat_right_side

        #test = self.point_2d_to_3d(point_2d, rotation_mat, intrinsic_matrix, translation_vector)

        return P.reshape(-1)


    # calculate a scale factor and a 3d point with Z=0, from a given 2d point, rotation matrix, intrinsic_matrix, and a translation vector:
        # calculate a scale factor and a 3d point with Z=0, from a given 2d point, rotation matrix, intrinsic_matrix, and a translation vector:
    # def point_2d_to_3d(self, point_2d, rotation_matrix, intrinsic_matrix, translation_vector):
    #     # calculate the scale factor:
    #     scale_factor = np.dot(np.dot(np.linalg.inv(intrinsic_matrix), rotation_matrix), translation_vector
    #                           ) / np.dot(np.dot(np.linalg.inv(intrinsic_matrix), rotation_matrix),
    #                                      np.array([point_2d[0], point_2d[1], 1]))
    #     # calculate the 3d point:
    #     point_3d = np.dot(np.dot(np.linalg.inv(intrinsic_matrix), rotation_matrix),
    #                       scale_factor * np.array([point_2d[0], point_2d[1], 1])) - translation_vector
    #     return point_3d


    # def calculate_scale_factor_and_3d_point(self, point_2d, rotation_matrix, translation_vector):
    #     # convert the point to homogeneous coordinates
    #     point_2d = np.array([point_2d[0], point_2d[1], 1], dtype=np.float32)
    #     # calculate the scale factor
    #     scale_factor = np.dot(rotation_matrix[2, :], translation_vector) / np.dot(rotation_matrix[2, :], point_2d)
    #     # calculate the 3d point
    #     point_3d = scale_factor * point_2d
    #     return point_3d

    # def image_point_to_3d(self, point_2d):
    #     # We assume that Z is 0, because our desk is at XY plane
    #     z_const = 0
    #
    #     # Transform 2d points to homogeneous coordinates
    #     point_2d = np.array([[point_2d[0], point_2d[1], 1]], dtype=np.float32).T
    #
    #     # Get rotation matrix
    #     rotation_mat = np.zeros(shape=(3, 3))
    #     cv2.Rodrigues(self.rotation_vector, rotation_mat)
    #     rotation_mat_inv = np.linalg.inv(rotation_mat)
    #
    #     # Get translation vector
    #     translation_vector, intrinsic_matrix = self.translation_vector.reshape(3, 1), self.intrinsic_mat
    #     intrinsic_matrix_inv = np.linalg.inv(intrinsic_matrix)
    #
    #     invR_invK_uv1 = rotation_mat_inv @ intrinsic_matrix_inv @ point_2d
    #     invR_tvec = rotation_mat_inv @ translation_vector
    #     s = (z_const + invR_tvec[2][0]) / invR_invK_uv1[2][0]
    #
    #     P = s * (invR_invK_uv1 - invR_tvec)
    #
    #     return P.reshape(-1)

    def set_graph_properties(self):
        self.graph.set_xlabel("X")
        self.graph.set_ylabel("Y")
        self.graph.set_zlabel("Z")
        self.graph.set_zlim(0, 2)

    def create_3d_graph(self):
        plt.clf()
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection='3d')
        self.graph = ax
        self.set_graph_properties()

    def add_graph_point(self, point_3d=None, color="y", full=False, points_list=None):
        if self.graph is None:
            self.create_3d_graph()
        if points_list is not None:
            self.graph.scatter(points_list[0], points_list[1], points_list[2], c=color)
        elif full:
            self.graph.scatter(point_3d[0], point_3d[1], point_3d[2], c=color)
        else:
            self.graph.scatter(point_3d[0], point_3d[1], point_3d[2], facecolors='none', edgecolors=color)


        # opengl_color = color
        # if isinstance(color, str):
        #     if color == "y":
        #         opengl_color = (239, 236, 0, 1)
        #     elif color == "b":
        #         opengl_color = (21, 82, 225, 1)
        #     elif color == "g":
        #         opengl_color = (23, 199, 0, 1)
        #     elif color == "r":
        #         opengl_color = (224, 20, 20, 1)
        #     opengl_color = tuple(ti / 255 if i < 3 else 1 for i, ti in enumerate(opengl_color))
        # else:
        #     opengl_color = opengl_color[0]
        # self.opengl_window.view.addItem(
        #     gl.GLScatterPlotItem(pos=point_3d.reshape(-1), color=opengl_color, size=.5, pxMode=False))

    graph_num_counter = 0

    def show_graph(self, save_fig=False, show_fig=False):
        if save_fig:
            plt.savefig(f'graphs/{Camera.graph_num_counter}.png')
            Camera.graph_num_counter += 1
        if show_fig:
            self.set_graph_properties()
            plt.show()

    def show_opengl_graph(self):
        self.opengl_window.show()
        self.opengl_app.exec()

    def save_graph(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.graph, f)

    def load_graph(self, path):
        with open(path, 'rb') as f:
            self.graph = pickle.load(f)
