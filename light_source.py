import glob
import pickle
from typing import List

import cv2
import numpy
import numpy as np
from sympy import Line3D, Point3D

import constants
import video_helper

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class LightSource:
    def __init__(self, camera):
        self.camera = camera
        self.pencil_len_mm = None
        self.light_pickle_name = 'light_calibration.pkl'
        self.points = []

    def calibrate(self, pencil_len_mm):
        def locate_points():
            return self.find_points()

        def set_calibration_parameters(params, to_pickle=False):
            self.points = params
            if to_pickle:
                with open(self.light_pickle_name, 'wb') as f2:
                    pickle.dump(params, f2)
            # Start calibration
            self.calibrate_helper()

        self.pencil_len_mm = pencil_len_mm

        if not constants.LIGHT_CALIBRATE_PICKLE:
            set_calibration_parameters(locate_points(), to_pickle=True)
        else:
            try:
                with open(self.light_pickle_name, 'rb') as f1:
                    print("Found light_calibrate.pkl")
                    set_calibration_parameters(pickle.load(f1))
            except (OSError, IOError):
                set_calibration_parameters(locate_points(), to_pickle=True)

    # def image_point_to_3d(self, point_2d):
    #
    #     # get rotation matrix
    #     # rotation_mat = np.zeros(shape=(3, 3))
    #     # cv2.Rodrigues(self.camera.rotation_vector, rotation_mat)
    #     # points_vector = np.array([[point_2d[0], point_2d[1], 1]], dtype=np.float32)
    #     # R_inv = np.linalg.inv(rotation_mat)
    #     # A_inv = np.linalg.inv(self.camera.intrinsic_mat)
    #     # uv_1 = points_vector
    #     # uv_1 = uv_1.T
    #     # suv_1 = 1 * uv_1
    #     # xyz_c = A_inv.dot(suv_1)
    #     # xyz_c = xyz_c - self.camera.translation_vector
    #     # point_3d = R_inv.dot(xyz_c)
    #
    #     rotation_mat = np.zeros(shape=(3, 3))
    #     cv2.Rodrigues(self.camera.rotation_vector, rotation_mat)
    #
    #     cam_center = -np.linalg.inv(rotation_mat.conj().T) @ self.camera.translation_vector.conj().T
    #
    #     extrinsic_mat = cv2.hconcat([rotation_mat, self.camera.translation_vector])
    #     projection_matrix = self.camera.intrinsic_mat @ extrinsic_mat
    #
    #     points_vector = np.array([[point_2d[0], point_2d[1], 1]], dtype=np.float32).T
    #
    #     point_3d = np.linalg.inv(self.camera.intrinsic_mat).dot(points_vector) - self.camera.translation_vector[2]
    #     point_3d = np.linalg.inv(rotation_mat).dot(point_3d)
    #
    #     # extrinsic_mat = cv2.hconcat([rotation_mat, self.camera.translation_vector])
    #     # projection_matrix = self.camera.intrinsic_mat @ extrinsic_mat
    #
    #     point_3d[2] = 0
    #     return point_3d.reshape(-1)

    def image_point_to_3d(self, point_2d):

        # We assume that Z is 0, because our desk is at XY plane
        z_const = 0

        # Transform 2d points to homogeneous coordinates
        point_2d = np.array([[point_2d[0], point_2d[1], 1]], dtype=np.float32).T

        # Get rotation matrix
        rotation_mat = np.zeros(shape=(3, 3))
        cv2.Rodrigues(self.camera.rotation_vector, rotation_mat)
        rotation_mat_inv = np.linalg.inv(rotation_mat)

        # Get translation vector
        translation_vector, intrinsic_matrix = self.camera.translation_vector.reshape(3, 1), self.camera.intrinsic_mat
        intrinsic_matrix_inv = np.linalg.inv(intrinsic_matrix)

        mat_left_side = rotation_mat_inv @ intrinsic_matrix_inv @ point_2d
        mat_right_side = rotation_mat_inv @ translation_vector

        # Find s:
        s = ((z_const + mat_right_side[2]) / mat_left_side[2])[0]

        # Calculate 3d points:
        P = (s * mat_left_side) - mat_right_side

        return P.reshape(-1)

    def calibrate_helper(self):

        # TODO: FIRST DO UNDISTORTION? (or not)

        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_zlim(0, 250)

        lines: List[Line3D] = []
        for b, ts in self.points:
            # convert image points to world points
            K = self.image_point_to_3d(b)
            TS = self.image_point_to_3d(ts)

            print(K)
            print(TS)

            # get the top point of the pencil
            T = (K[0], K[1], K[2] + self.pencil_len_mm)

            # calculate the direction from ts to pencil top
            line_direction = np.subtract(T, TS)

            # create a line from ts and the direction
            #lines.append(Line3D(TS, direction_ratio=line_direction))
            lines.append(Line3D(Point3D(TS), Point3D(T)))


            # --- PLOTS ---
            ax.scatter(K[0], K[1], K[2], facecolors='none', edgecolors='r')  # plot the point K on the figure
            ax.scatter(TS[0], TS[1], TS[2], facecolors='none', edgecolors='b')  # plot the point K on the figure
            ax.scatter(T[0], T[1], T[2], facecolors='none', edgecolors='y')  # plot the point K on the figure

            # plot the line between K and TS
            x, y, z = [K[0], TS[0]], [K[1], TS[1]], [K[2], TS[2]]
            ax.plot(x, y, z, color='g')

            # plot the line between K and T
            x, y, z = [K[0], T[0]], [K[1], T[1]], [K[2], T[2]]
            ax.plot(x, y, z, color='y')

            t = 3  # length of the line
            T_TS_Line = TS + (t * line_direction)  # T-TS line

            x = [TS[0], T[0], T_TS_Line[0]]
            y = [TS[1], T[1], T_TS_Line[1]]
            z = [TS[2], T[2], T_TS_Line[2]]
            # Plotting the line
            plt.plot(x, y, z, 'r', linewidth=2)

        plt.show()

        # find the intersection point of the lines and then take the average point
        intersection1 = lines[0].intersection(lines[1])
        intersection2 = lines[0].intersection(lines[2])

        if intersection1 or intersection2:
            print("There is an intersection!")

        if not intersection1:
            print("No intersection_1 found => EXIT")
            return
        if not intersection2:
            print("No intersection_2 found => EXIT")
            return

        # NOTE: sympy represents number as expressions, so to get the decimal number,
        # convert to float

        a = [intersection1[0].x, intersection1[0].y, intersection1[0].z]
        b = [intersection2[0].x, intersection2[0].y, intersection2[0].z]
        light_source_pos = np.average([a, b], axis=0)
        print(light_source_pos)
        pass

    def find_points(self):
        videos = glob.glob('light_calibration/*.mp4')
        points = []
        count = 0
        for video_name in videos:
            frame = video_helper.get_frame_from_video(video_name=video_name, frame_number=0)
            cv2.imwrite(f"./light_calibration_images/{count}.jpg", frame)
            count += 1
            chosen_coordinates = []
            self.show_pixel_selection(frame=frame, click_callback=lambda x, y: chosen_coordinates.append((x, y)))
            self.show_pixel_selection(frame=frame, click_callback=lambda x, y: chosen_coordinates.append((x, y)))
            b, ts = chosen_coordinates[0], chosen_coordinates[1]
            print(b, ts)
            points.append((b, ts))
        return points

    def show_pixel_selection(self, frame: np.ndarray, click_callback):
        def on_click(event):
            if event.dblclick:
                click_callback(event.xdata, event.ydata)
                plt.close()

        fig = plt.figure()
        fig.canvas.mpl_connect('button_press_event', on_click)
        plt.imshow(frame)
        plt.show()
