import glob
import pickle

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

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

    def find_points(self):
        videos = glob.glob('light_calibration/*.mp4')
        points = []
        for video_name in videos:
            frame = video_helper.get_frame_from_video(video_name=video_name, frame_number=0)
            chosen_coordinates = []
            self.show_pixel_selection(frame=frame, click_callback=lambda x, y: chosen_coordinates.append((x, y)))
            self.show_pixel_selection(frame=frame, click_callback=lambda x, y: chosen_coordinates.append((x, y)))
            b, ts = chosen_coordinates[0], chosen_coordinates[1]
            points.append((b, ts))
        return points

    def calculate_XYZ(self, u, v):
        ## MAYBE USE THE REAL WORLD POINTS TO CALCULATE THE PLANE OF THE DESK

        # Solve: From Image Pixels, find World Points
        uv_1 = np.array([[u, v, 1]], dtype=np.float32)
        uv_1 = uv_1.T
        suv_1 = self.scalingfactor * uv_1
        xyz_c = self.inverse_newcam_mtx.dot(suv_1)
        xyz_c = xyz_c - self.tvec1
        XYZ = self.inverse_R_mtx.dot(xyz_c)
        return XYZ

    def calibrate_helper(self):
        # TODO: FIRST DO UNDISTORTION
        #print(self.points)
        pass

    def show_pixel_selection(self, frame: np.ndarray, click_callback):
        def on_click(event):
            if event.dblclick:
                click_callback(event.x, event.y)
                plt.close()

        fig = plt.figure()
        fig.canvas.mpl_connect('button_press_event', on_click)
        plt.imshow(frame)
        plt.show()
