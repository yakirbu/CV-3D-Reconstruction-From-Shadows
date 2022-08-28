import glob
import pickle
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sympy import Line3D, Point3D

import constants
import video_helper


class LightSource:
    def __init__(self, camera, pencil_len_mm=126):
        self.camera = camera
        self.pencil_len_mm = pencil_len_mm
        self.light_pickle_name = 'light_calibration.pkl'
        self.points = []

    def calibrate(self):
        def locate_points():
            return self.find_points()

        def set_calibration_parameters(params, to_pickle=False):
            self.points = params
            if to_pickle:
                with open(self.light_pickle_name, 'wb') as f2:
                    pickle.dump(params, f2)
            # Start calibration
            self.calibrate_helper()

        if not constants.LIGHT_CALIBRATE_PICKLE:
            set_calibration_parameters(locate_points(), to_pickle=True)
        else:
            try:
                with open(self.light_pickle_name, 'rb') as f1:
                    print("Found light_calibrate.pkl")
                    set_calibration_parameters(pickle.load(f1))
            except (OSError, IOError):
                set_calibration_parameters(locate_points(), to_pickle=True)

    def image_point_to_3d(self, point_2d):

        # get rotation matrix
        rotation_mat = np.zeros(shape=(3, 3))
        cv2.Rodrigues(self.camera.rotation_vector, rotation_mat)

        extrinsic_mat = cv2.hconcat([rotation_mat, self.camera.translation_vector])
        projection_matrix = self.camera.intrinsic_mat @ extrinsic_mat

        points_vector = np.array([[point_2d[0], point_2d[1], 1]], dtype=np.float32).T

        point_3d = np.linalg.inv(self.camera.intrinsic_mat).dot(points_vector) - self.camera.translation_vector
        point_3d = np.linalg.inv(rotation_mat).dot(point_3d)
        return projection_matrix @ np.append(point_3d, 1)

    def calibrate_helper(self):

        # TODO: FIRST DO UNDISTORTION? (or not)

        lines: List[Line3D] = []
        for b, ts in self.points:

            # convert image points to world points
            K = self.image_point_to_3d(b)
            TS = self.image_point_to_3d(ts)

            # get the top point of the pencil
            pencil_top = (K[0], K[1] + self.pencil_len_mm, 1)

            # calculate the direction from ts to pencil top
            line_direction = np.subtract(pencil_top, TS)

            # create a line from ts and the direction
            lines.append(Line3D(TS, direction_ratio=line_direction))

        # find the intersection point of the lines and then take the average point
        intersection1 = lines[0].intersection(lines[1])
        intersection2 = lines[0].intersection(lines[2])

        # NOTE: sympy represents number as expressions, so to get the decimal number,
        # convert to float
        light_source_pos = np.divide((intersection1 + intersection2), 2)
        print(light_source_pos)
        pass

    def find_points(self):
        videos = glob.glob('light_calibration/*.mp4')
        points = []
        for video_name in videos:
            frame = video_helper.get_frame_from_video(video_name=video_name, frame_number=0)
            chosen_coordinates = []
            self.show_pixel_selection(frame=frame, click_callback=lambda x, y: chosen_coordinates.append((x, y)))
            self.show_pixel_selection(frame=frame, click_callback=lambda x, y: chosen_coordinates.append((x, y)))
            b, ts = chosen_coordinates[0], chosen_coordinates[1]
            print(b,ts)
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
