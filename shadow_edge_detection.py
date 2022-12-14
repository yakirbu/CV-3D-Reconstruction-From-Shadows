import glob
import pickle

import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import constants
import utils
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
        self.first_frame = None

    def set_fixed_points(self, fixed_points):
        """
        In contrast to the article, here we set a fixed x_left and x_right instead of y_top and y_bot.
        :param x:
        :param is_right:
        :return:
        """
        self.x_right, self.x_left = int(fixed_points[0]), int(fixed_points[1])
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
        video_helper.show_pixel_selection(frame=frame,
                                          click_callback=lambda x, y: chosen_coordinates.append((x, y)),
                                          title="Select x_right", is_gray=True)
        video_helper.show_pixel_selection(frame=frame,
                                          click_callback=lambda x, y: chosen_coordinates.append((x, y)),
                                          title="Select x_left", is_gray=True)
        return chosen_coordinates[0][0], chosen_coordinates[1][0]

    def detect(self):
        pixels_intensities = {}

        # Get the first colored frame of the video
        self.first_frame = self.get_colored_first_frame()

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
            # Save images
            video_helper.save_image(frame, f"./shadow_images/3d_scan_loop")

        # print (pixels_intensities)
        for pixel, intensity in pixels_intensities.copy().items():
            if pixel[0] == self.x_right or pixel[0] == self.x_left:
                continue
            if intensity.max_intensity - intensity.min_intensity < constants.INTENSITY_THRESHOLD:
                del pixels_intensities[pixel]

        self.pixels_intensities = pixels_intensities

        spatial_edge_pixels_list = []
        break_occurred = False
        for frame in self.load_frames():
            video_helper.save_image(self.get_filtered_image(frame), f"./shadow_images/filtered_scan")
            spatial_edge = self.analyze_frame(frame)
            if spatial_edge is None:
                break_occurred = True
                break
            spatial_edge_pixels_list.append(spatial_edge)

        self.triangulation(spatial_edge_pixels_list)

        # If everything went fine, save the graph:
        if not break_occurred:
            self.camera.save_graph(constants.PICKLE_3D_OBJECT_NAME)

        self.camera.show_opengl_graph()

    def analyze_frame(self, frame):
        """
        Analyze the frame and return the shadowed pixels
        :param frame:
        :return:
        """

        A_t, B_t = self.handle_fixed_points(frame=frame)

        # save frame:
        video_helper.save_image(frame, f"./shadow_images/x_left_x_right")

        spatial_edge_pixels = []
        for pixel, pixel_data in self.pixels_intensities.copy().items():
            dI = self.get_pixel(frame, pixel[0], pixel[1]) - pixel_data.get_intensity_avg()
            if pixel_data.last_diff_intensity and \
                    (pixel_data.last_diff_intensity < 0 < dI or pixel_data.last_diff_intensity > 0 > dI):
                # Zero crossing

                # Ignore x_left and x_right pixels, since we handle them differently
                if pixel[0] == self.x_right or pixel[0] == self.x_left:
                    continue

                # # If the pixel is zero crossing, delete it from the dictionary
                # del self.pixels_intensities[pixel]
                # # mark it in the image
                # self.set_pixel(frame, pixel[0], pixel[1], 255)
                # add it to the list of edge-shadowed pixels
                spatial_edge_pixels.append(pixel)

            pixel_data.last_diff_intensity = dI

        if utils.is_valid(spatial_edge_pixels):
            y_avg = sum([pixel[1] for pixel in spatial_edge_pixels]) / len(spatial_edge_pixels)
            y_threshold = 0.1 * frame.shape[0]
            # remove pixels that are too far from the average y:
            spatial_edge_pixels = [pixel for pixel in spatial_edge_pixels if abs(pixel[1] - y_avg) < y_threshold]
            for pixel in spatial_edge_pixels:
                # If the pixel is zero crossing, delete it from the dictionary
                del self.pixels_intensities[pixel]
                # mark it in the image
                self.set_pixel(frame, pixel[0], pixel[1], 255)


        # self.triangulation(spatial_edge_pixels=spatial_edge_pixels, P1=A_t, P2=B_t, S=self.light.light_position)

        # self.get_filtered_image(frame)

        video_helper.save_image(frame, f"./shadow_images/spatial_edges")

        # if video_helper.imshow(frame, to_wait=False):
        #     return None
        return spatial_edge_pixels, A_t, B_t

    # def triangulation(self, spatial_edge_pixels, P1, P2, S):
    #     if not utils.is_valid(spatial_edge_pixels) or not utils.is_valid(P1) or not utils.is_valid(
    #             P2) or not utils.is_valid(S):
    #         print("Invalid data (happens usually on the first frame)")
    #         return
    #
    #     C = self.camera.cam_center
    #
    #     # stack P1, P2, S and C horizontally to create a matrix of 3x3
    #     A = np.hstack((S, P1, P2, C))
    #     A = np.vstack(([1, 1, 1, 1], A))
    #
    #     det_A = np.linalg.det(A)
    #
    #     P_points = []
    #
    #     for pixel in spatial_edge_pixels:
    #         if not self.x_left <= pixel[0] <= self.x_right:
    #             continue
    #
    #         P0 = self.camera.image_point_to_3d(pixel).reshape(-1, 1)
    #
    #         B = np.hstack((S, P1, P2, np.subtract(P0, C)))
    #         B = np.vstack(([1, 1, 1, 0], B))
    #
    #         det_B = np.linalg.det(B)
    #         t = -(det_A / det_B)
    #
    #         line_direction = np.subtract(P0, C)
    #         P = C + (t * line_direction)
    #
    #         P_points.append([pixel, P])
    #
    #         # if not (0 <= P[2] <= 1):
    #         #     continue
    #         #
    #         # # Color the point P with its original color from the first frame
    #         # b, g, r = self.get_pixel(self.first_frame, pixel[0], pixel[1])
    #         # self.camera.add_graph_point(point_3d=P, color=[(r / 255, g / 255, b / 255, 1)], full=True)
    #
    #     P_points = np.array(P_points)
    #
    #     x_avg = np.mean(P_points[:, 1][:][0][0])
    #     y_avg = np.mean(P_points[:, 1][:][0][1])
    #     z_avg = np.mean(P_points[:, 1][:][0][2])
    #     avg_threshold = 1.5
    #     for p in P_points:
    #         if abs(p[1][0] - x_avg) < avg_threshold and abs(p[1][1] - y_avg) < avg_threshold and abs(
    #                 p[1][2] - z_avg) < avg_threshold:
    #             # Color the point P with its original color from the first frame
    #             b, g, r = self.get_pixel(self.first_frame, p[0][0], p[0][1])
    #             self.camera.add_graph_point(point_3d=p[1], color=[(r / 255, g / 255, b / 255, 1)], full=True)
    #
    #     self.camera.show_graph(save_fig=True, show_fig=False)

    def get_Ps(self, spatial_edge_pixels, P1, P2):

        S, C = self.light.light_position, self.camera.cam_center

        if not utils.is_valid(spatial_edge_pixels) or not utils.is_valid(P1) or not utils.is_valid(
                P2) or not utils.is_valid(S):
            print("Invalid data (happens usually on the first frame)")
            return []

        # stack P1, P2, S and C horizontally to create a matrix of 3x3
        A = np.hstack((S, P1, P2, C))
        A = np.vstack(([1, 1, 1, 1], A))

        det_A = np.linalg.det(A)

        P_points = []

        for pixel in spatial_edge_pixels:
            if not self.x_left <= pixel[0] <= self.x_right:
                continue

            P0 = self.camera.image_point_to_3d(pixel).reshape(-1, 1)

            B = np.hstack((S, P1, P2, np.subtract(P0, C)))
            B = np.vstack(([1, 1, 1, 0], B))

            det_B = np.linalg.det(B)
            t = -(det_A / det_B)

            line_direction = np.subtract(P0, C)
            P = C + (t * line_direction)

            P_points.append([pixel, P])
        return P_points

    def triangulation(self, spatial_edge_pixels_list):
        P_s = np.asarray([p for edge in spatial_edge_pixels_list for p in self.get_Ps(edge[0], edge[1], edge[2])])

        # from the tuple pixel, point in P_s, get the average of the x, y, z coordinates of all the points in the list
        x_avg = np.mean([np.mean(p[1][0]) for p in P_s])
        y_avg = np.mean([np.mean(p[1][1]) for p in P_s])
        z_avg = np.mean([np.mean(p[1][2]) for p in P_s])

        avg_threshold = 20
        z_threshold = 1

        counter = 1
        for p in P_s:
            if not utils.is_valid(p):
                continue
            x, y, z = p[1][0][0], p[1][1][0], p[1][2][0]
            if True: #abs(x - x_avg) < avg_threshold and abs(y - y_avg) < avg_threshold and abs(z - z_avg) < z_threshold:

                # Color the point P with its original color from the first frame
                b, g, r = self.get_pixel(self.first_frame, p[0][0], p[0][1])
                color = (r / 255, g / 255, b / 255, 1)
                self.camera.add_graph_point(point_3d=[x, y, z], color=[color], full=True)
                if counter % 1000 == 0:
                    # self.camera.add_graph_point(points_list=(x_list, y_list, z_list), color=colors, full=True)
                    self.camera.show_graph(save_fig=True, show_fig=False)
                counter += 1
        self.camera.show_graph(save_fig=True, show_fig=False)

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
                        p_data.max_intensity - p_data.min_intensity) > constants.INTENSITY_THRESHOLD:

                    # for i in range(1, 3):
                    #     self.set_pixel(frame, x + i, y, 255)
                    #     self.set_pixel(frame, x - i, y, 255)
                    #     self.set_pixel(frame, x + i, y + i, 255)
                    #     self.set_pixel(frame, x - i, y - i, 255)
                    #     self.set_pixel(frame, x, y + i, 255)
                    #     self.set_pixel(frame, x, y - i, 255)
                    # self.set_pixel(frame, x, y, 255)
                    self.set_pixel(frame, x, y, 255)
                    return x, y

    def handle_fixed_points(self, frame):
        # self.camera.create_3d_graph()
        cord_1 = self.find_y_left_right(frame, is_right=True)
        cord_2 = self.find_y_left_right(frame, is_right=False)

        if not cord_1 or not cord_2:
            return None, None

        self.y_right = cord_1[1]
        self.y_left = cord_2[1]

        B_t = self.camera.image_point_to_3d((self.x_left, self.y_left))
        A_t = self.camera.image_point_to_3d((self.x_right, self.y_right))
        S = self.light.light_position.reshape(-1)

        # print(f"B_t: {B_t}, A_t: {A_t}, S: {S}")

        # self.camera.add_graph_point(point_3d=B_t, color='r', full=False)
        # self.camera.add_graph_point(point_3d=A_t, color='r', full=False)
        # self.camera.add_graph_point(point_3d=S, color='y', full=True)

        # # 1. create vertices from points
        # verts = [list(zip([A_t[0], B_t[0], S[0]], [A_t[1], B_t[1], S[1]], [A_t[2], B_t[2], S[2]]))]
        # # 2. create 3d polygons and specify parameters
        # srf = Poly3DCollection(verts, alpha=.25, facecolor='#800000')
        # # 3. add polygon to the figure (current axes)
        # plt.gca().add_collection3d(srf)

        return A_t.reshape(-1, 1), B_t.reshape(-1, 1)

    def get_filtered_image(self, frame):
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

        # video_helper.imshow(frame, to_wait=True)
        return frame

    def load_frames(self, grayscale=True):
        return video_helper.generate_frames(glob.glob("./shadow_edge_detection/*.mp4")[0], grayscale=grayscale)

    def get_colored_first_frame(self):
        return video_helper.get_frame_from_video(glob.glob("./shadow_edge_detection/*.mp4")[0], frame_number=0)
