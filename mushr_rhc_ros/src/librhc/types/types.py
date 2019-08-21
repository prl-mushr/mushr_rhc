# Copyright (c) 2019, The Personal Robotics Lab, The MuSHR Team, The Contributors of MuSHR
# License: BSD 3-Clause. See LICENSE.md file in root directory.

import numpy as np


class MapData:
    def __init__(
        self,
        name,
        resolution,
        origin_x,
        origin_y,
        orientation_angle,
        width,
        height,
        get_map_data,
    ):
        self.name = name
        self.resolution = resolution
        self.origin_x, self.origin_y = origin_x, origin_y
        self.angle = orientation_angle
        self.angle_sin, self.angle_cos = np.sin(self.angle), np.cos(self.angle)
        self.width = width
        self.height = height
        self._data = None
        self._get_map_data = get_map_data

    def data(self):
        if not self._data:
            self._data = np.array(self._get_map_data())
        return self._data
