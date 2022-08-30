import glob
import pickle

import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import constants
import video_helper
from pixel import Pixel


class ShadowEdgeDetection:
    def __init__(self, camera, light):
        self.shadow_pickle_name = "shadow_fixed_points.pkl"
        self.camera = camera
        self.light = light
        self.pixels_intensities = {}
        self.x_left = 0
        self.x_right = 0
        self.y_left = 0
        self.y_right = 0

    def set_fixed_points(self, fixed_points):
        """
        In contrast to the article, here we set a fixed x_left and x_right instead of y_top and y_bot.
        :param x:
        :param is_right:
        :return:
        """
        self.x_right, self.x_left = int(fixed_points[0]), int(fixed_points[1])
        if constants.SHADOW_POINTS_PICKLE:
            with open(self.shadow_pickle_name, 'wb') as f1:
                pickle.dump((self.x_right, self.x_left), f1)
        # print x_right and x_left:
        print(f"x_right: {self.x_right}, x_left: {self.x_left}")

    def select_fixed_points(self, frame):
        """
        Select the fixed points for the shadow edge detection algorithm
        :param frame:
        :return:
        """
        chosen_coordinates = []
        video_helper.show_pixel_selection(camera=self.camera,
                                          frame=frame,
                                          click_callback=lambda x, y: chosen_coordinates.append((x, y)),
                                          title="Select x_right", is_gray=True)
        video_helper.show_pixel_selection(camera=self.camera,
                                          frame=frame,
                                          click_callback=lambda x, y: chosen_coordinates.append((x, y)),
                                          title="Select x_left", is_gray=True)
        return chosen_coordinates[0][0], chosen_coordinates[1][0]

    def detect(self):
        pixels_intensities = {}

        for k, frame in enumerate(self.load_frames()):
            for y in range(frame.shape[0]):
                for x in range(frame.shape[1]):
                    frame_intensity = self.get_pixel(frame, x, y)
                    if (x, y) not in pixels_intensities:
                        pixels_intensities[(x, y)] = Pixel(min_intensity=frame_intensity, max_intensity=frame_intensity)
                    else:
                        if frame_intensity < pixels_intensities[(x, y)].min_intensity:
                            pixels_intensities[(x, y)].min_intensity = frame_intensity
                        if frame_intensity > pixels_intensities[(x, y)].max_intensity:
                            pixels_intensities[(x, y)].max_intensity = frame_intensity
            if k == 0:
                if not constants.SHADOW_POINTS_PICKLE:
                    self.set_fixed_points(self.select_fixed_points(frame))
                else:
                    try:
                        with open(self.shadow_pickle_name, 'rb') as f1:
                            print(f"Found {self.shadow_pickle_name}")
                            self.set_fixed_points(pickle.load(f1))
                    except (OSError, IOError):
                        self.set_fixed_points(self.select_fixed_points(frame))

        # print (pixels_intensities)
        for pixel, intensity in pixels_intensities.copy().items():
            if pixel[0] == self.x_right or pixel[0] == self.x_left:
                continue
            if intensity.max_intensity - intensity.min_intensity < constants.INTENSITY_TRESHOLD:
                del pixels_intensities[pixel]

        self.pixels_intensities = pixels_intensities

        for frame in self.load_frames():
            if not self.analyze_frame(frame):
                break

    def get_pixel(self, frame, x, y):
        return frame[y][x]

    def set_pixel(self, frame, x, y, value):
        frame[y][x] = value

    def find_y_left_right(self, frame, is_right):
        x = self.x_right if is_right else self.x_left
        for y, pixel_intensity in enumerate(frame[:, x]):
            p_data = self.pixels_intensities[(x, y)]
            dI = pixel_intensity - p_data.get_intensity_avg()
            p_data.last_dI = dI
            if p_data.last_diff_intensity and \
                    (p_data.last_diff_intensity < 0 < dI or p_data.last_diff_intensity > 0 > dI):
                # zero crossing
                p_data.zero_crossing_times += 1
                if p_data.zero_crossing_times == 1 and (
                        p_data.max_intensity - p_data.min_intensity) > constants.INTENSITY_TRESHOLD:
                    self.set_pixel(frame, x, y, 255)
                    return x, y

    def analyze_frame(self, frame):
        """
        Analyze the frame and return the shadowed pixels
        :param frame:
        :return:
        """

        plt.clf()

        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_zlim(0, 1000)

        cord_1 = self.find_y_left_right(frame, is_right=True)
        cord_2 = self.find_y_left_right(frame, is_right=False)
        if cord_1 and cord_2:
            self.y_right = cord_1[1]
            self.y_left = cord_2[1]

            # Plot
            # x, y = [self.x_left, self.x_right], [self.y_left, self.y_right]
            # ax.plot(x, y, color='y')
            # ax.imshow(frame, cmap='gray')

            B_t = self.camera.image_point_to_3d((self.x_left, self.y_left))
            A_t = self.camera.image_point_to_3d((self.x_right, self.y_right))
            S = self.light.light_position

            print(f"B_t: {B_t}, A_t: {A_t}, S: {S}")

            # u = S - B_t
            # v = S - A_t
            # normal = np.cross(u, v)
            # d = -np.dot(normal, S)
            # xx, yy = np.meshgrid(range(30), range(30))
            # a, b, c = normal
            # z = (d - a * xx - b * yy) / c
            # ax.plot_surface(xx, yy, z)

            # 1. create vertices from points
            verts = [list(zip([A_t[0], B_t[0], S[0]], [A_t[1], B_t[1], S[1]], [A_t[2], B_t[2], S[2]]))]
            # 2. create 3d polygons and specify parameters
            srf = Poly3DCollection(verts, alpha=.25, facecolor='#800000')
            # 3. add polygon to the figure (current axes)
            plt.gca().add_collection3d(srf)

            ax.scatter(B_t[0], B_t[1], B_t[2], facecolors='none', edgecolors='r')  # plot the point K on the figure
            ax.scatter(A_t[0], A_t[1], A_t[2], facecolors='none', edgecolors='r')  # plot the point K on the figure
            ax.scatter(S[0], S[1], S[2], facecolors='none', edgecolors='y')  # plot the point K on the figure

            #plt.show()

        # if video_helper.imshow(frame, to_wait=True):
        #     return False

        for pixel, pixel_data in self.pixels_intensities.copy().items():
            dI = self.get_pixel(frame, pixel[0], pixel[1]) - pixel_data.get_intensity_avg()
            if pixel_data.last_diff_intensity and \
                    (pixel_data.last_diff_intensity < 0 < dI or pixel_data.last_diff_intensity > 0 > dI):
                # Zero crossing

                # Ignore x_left and x_right pixels, since we handle them differently
                if pixel[0] == self.x_right or pixel[0] == self.x_left:
                    continue

                # If the pixel is zero crossing, delete it from the dictionary
                del self.pixels_intensities[pixel]
                # mark it in the image
                self.set_pixel(frame, pixel[0], pixel[1], 0)

            pixel_data.last_diff_intensity = dI

        # self.show_shadow_pixels(frame)

        if video_helper.imshow(frame, to_wait=True):
            return False
        return True

    def show_shadow_pixels(self, frame):
        """
        Show the pixels that are shadowed in the frame
        :param frame:
        :return:
        """
        frame = frame.copy()
        for y in range(frame.shape[0]):
            for x in range(frame.shape[1]):
                if (x, y) not in self.pixels_intensities:
                    self.set_pixel(frame, x, y, 0)

        video_helper.imshow(frame, to_wait=True)

    def load_frames(self):
        return video_helper.generate_frames(glob.glob("./shadow_edge_detection/*.mp4")[0],
                                            resize_factor=constants.SHADOW_RESIZE_FACTOR,
                                            grayscale=True)
