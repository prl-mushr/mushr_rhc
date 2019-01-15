import numpy as np
import md5


class MapData:
    def __init__(self, resolution, origin_x, origin_y, orientation_angle, width, height, data):
        self.resolution = resolution
        self.origin_x, self.origin_y = origin_x, origin_y
        self.angle = orientation_angle
        self.angle_sin, self.angle_cos = np.sin(self.angle), np.cos(self.angle)
        self.width = width
        self.height = height
        self.data = np.array(data)

        datamd5 = md5.new("".join(map(str, self.data)))
        self.data_md5 = datamd5.hexdigest()
