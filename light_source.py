import glob
import pickle
import cv2
import numpy as np
import constants
import video_helper
import matplotlib.pyplot as plt

class LightSource:
    def __init__(self, camera):
        self.camera = camera
        self.pencil_len_mm = None
        self.light_pickle_name = 'light_calibration.pkl'
        self.points = []
        self.light_position = None

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

    def calibrate_helper(self, height_of_lamp=800):

        # TODO: FIRST DO UNDISTORTION? (or not)

        # fig = plt.figure(figsize=(4, 4))
        # ax = fig.add_subplot(111, projection='3d')
        # ax.set_xlabel("X")
        # ax.set_ylabel("Y")
        # ax.set_zlabel("Z")
        # ax.set_zlim(0, height_of_lamp)

        T_points = []
        TS_points = []

        for b, ts in self.points:
            # convert image points to world points
            K = self.camera.image_point_to_3d(b)
            TS = self.camera.image_point_to_3d(ts)

            print(f"Calculating for points: b={b}, ts={ts}")

            # get the top point of the pencil
            T = (K[0], K[1], K[2] + self.pencil_len_mm)

            T_points.append(T)
            TS_points.append(TS)

            if constants.LIGHT_CALIBRATE_GRAPHS:
                # --- PLOTS ---
                self.camera.add_graph_point(K, 'r')  # plot the point K on the figure
                self.camera.add_graph_point(TS, 'b')  # plot the point K on the figure
                self.camera.add_graph_point(T, 'y')  # plot the point K on the figure

                # plot the line between K and TS
                x, y, z = [K[0], TS[0]], [K[1], TS[1]], [K[2], TS[2]]
                self.camera.graph.plot(x, y, z, color='g')

                # plot the line between K and T
                x, y, z = [K[0], T[0]], [K[1], T[1]], [K[2], T[2]]
                self.camera.graph.plot(x, y, z, color='y')

                t = height_of_lamp/100  # length of the line
                line_direction = np.subtract(T, TS)
                T_TS_Line = TS + (t * line_direction)  # T-TS line

                x = [TS[0], T[0], T_TS_Line[0]]
                y = [TS[1], T[1], T_TS_Line[1]]
                z = [TS[2], T[2], T_TS_Line[2]]
                # Plotting the line
                self.camera.graph.plot(x, y, z, 'r', linewidth=2)

        # Reshape points
        TS_points, T_points = np.array(TS_points).reshape((-1, 3)), np.array(T_points).reshape((-1, 3))
        intersection_point = LightSource.find_closet_intersection_point(T_points, TS_points)

        if constants.LIGHT_CALIBRATE_GRAPHS:
            # Add the point to the graph
            self.camera.add_graph_point(intersection_point, color='y')
            #plt.show()

        self.light_position = intersection_point

    @staticmethod
    def find_closet_intersection_point(points_1, points_2):
        """
        Find the closest intersection point between multiple lines
        :param points_1:
        :param points_2:
        :return:
        """
        # generate all line direction vectors
        n = (points_1 - points_2) / np.linalg.norm(points_1 - points_2, axis=1)[:, np.newaxis]  # normalized
        # generate the array of all projectors
        projs = np.eye(n.shape[1]) - n[:, :, np.newaxis] * n[:, np.newaxis]  # I - n*n.T
        # generate R matrix and q vector
        R = projs.sum(axis=0)
        q = (projs @ points_2[:, :, np.newaxis]).sum(axis=0)
        # solve the least squares problem for the intersection point p: Rp = q
        p = np.linalg.lstsq(R, q, rcond=None)[0]
        return p.reshape(-1)

    def find_points(self):
        videos = glob.glob('light_calibration/*.mp4')
        points = []
        count = 0
        for video_name in videos:
            frame = video_helper.get_frame_from_video(video_name=video_name, frame_number=0)
            cv2.imwrite(f"./light_calibration_images/{count}.jpg", frame)
            count += 1
            chosen_coordinates = []
            video_helper.show_pixel_selection(camera=self.camera,
                                              frame=frame,
                                              click_callback=lambda x, y: chosen_coordinates.append((x, y)),
                                              title="Select the bottom point of the pencil")
            video_helper.show_pixel_selection(camera=self.camera,
                                              frame=frame,
                                              click_callback=lambda x, y: chosen_coordinates.append((x, y)),
                                              title="Select top point of the shadow")
            b, ts = chosen_coordinates[0], chosen_coordinates[1]
            print(f"Chosen points: b={b}, ts={ts}")
            points.append((b, ts))
        return points

