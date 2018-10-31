import numpy as np

class MapData:
    def __init__(self, resolution, origin_x, origin_y, orientation_angle, width, height, data):
        self.resolution = resolution
        self.origin_x, self.origin_y = origin_x, origin_y
        self.angle = orientation_angle
        self.angle_sin, self.angle_cos = np.cos(self.angle), np.sin(self.angle)
        self.width = width
        self.height = height
        self.data = data
