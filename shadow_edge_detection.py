import glob

import cv2
import numpy as np
from matplotlib import pyplot as plt

import constants
import video_helper
from pixel import Pixel


class ShadowEdgeDetection:
    def __init__(self, camera, light):
        self.camera = camera
        self.light = light
        self.pixels_intensities = {}

    def detect(self):
        pixels_intensities = {}

        for frame in self.load_frames():
            for i in range(frame.shape[0]):
                for j in range(frame.shape[1]):
                    frame_intensity = frame[i, j]
                    if (i, j) not in pixels_intensities:
                        pixels_intensities[(i, j)] = Pixel(min_intensity=frame_intensity, max_intensity=frame_intensity)
                    else:
                        if frame_intensity < pixels_intensities[(i, j)].min_intensity:
                            pixels_intensities[(i, j)].min_intensity = frame_intensity
                        if frame_intensity > pixels_intensities[(i, j)].max_intensity:
                            pixels_intensities[(i, j)].max_intensity = frame_intensity

        # print (pixels_intensities)
        for pixel, intensity in pixels_intensities.copy().items():
            if intensity.max_intensity - intensity.min_intensity < constants.INTENSITY_TRESHOLD:
                del pixels_intensities[pixel]

        self.pixels_intensities = pixels_intensities

        for frame in self.load_frames():
            for pixel, pixel_data in self.pixels_intensities.copy().items():
                dI = frame[pixel[0], pixel[1]] - pixel_data.get_intensity_avg()
                print(dI)

                if pixel_data.last_diff_intensity and \
                        (pixel_data.last_diff_intensity < 0 < dI or pixel_data.last_diff_intensity > 0 > dI):

                    print("Zero crossing")
                    del self.pixels_intensities[pixel]
                    frame[pixel[0], pixel[1]] = 255
                pixel_data.last_diff_intensity = dI
            #self.show_shadow_pixels(frame)
            cv2.imshow("frame", frame)
            cv2.waitKey()

    def show_shadow_pixels(self, frame):
        """
        Show the pixels that are shadowed in the frame
        :param frame:
        :return:
        """
        frame = frame.copy()
        for i in range(frame.shape[0]):
            for j in range(frame.shape[1]):
                if (i, j) not in self.pixels_intensities:
                    frame[i, j] = 0

        cv2.imshow("frame", frame)
        cv2.waitKey()

    def load_frames(self):
        return video_helper.generate_frames(glob.glob("./shadow_edge_detection/*.mp4")[0],
                                            resize_factor=constants.SHADOW_RESIZE_FACTOR,
                                            grayscale=True)
