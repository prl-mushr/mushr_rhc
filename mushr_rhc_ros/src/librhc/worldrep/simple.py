# Copyright (c) 2019, The Personal Robotics Lab, The MuSHR Team, The Contributors of MuSHR
# License: BSD 3-Clause. See LICENSE.md file in root directory.

import numpy as np
import torch
from scipy import ndimage

import librhc.utils as utils
import rhctensor


class Simple:
    def __init__(self, params, logger, dtype, map):
        """
        Inputs:
            params (obj): parameter getter
            logger (obj): a logger to write to
            dtype (obj): data type for tensors
            map (types.MapData): A map representation
        """
        self.params = params
        self.logger = logger
        self.dtype = dtype
        self.map = map
        self.perm_reg = utils.map.load_permissible_region(self.params, self.map)

        self.reset()

    def reset(self):
        self.T = self.params.get_int("T", default=15)
        self.K = self.params.get_int("K", default=62)
        self.epsilon = self.params.get_float("world_rep/epsilon", default=0.5)

        self.scaled = self.dtype(self.K * self.T, 3)
        self.bbox_map = self.dtype(self.K * self.T, 2, 4)
        self.perm = rhctensor.byte_tensor()(self.K * self.T)

        # Ratio of car to extend in every direction
        self.car_length = self.params.get_float("world_rep/car_length", default=0.5)
        self.car_width = self.params.get_float("world_rep/car_width", default=0.3)

        self.dist_field = ndimage.distance_transform_edt(
            np.logical_not(self.perm_reg.cpu().numpy())
        )

        self.dist_field *= self.map.resolution
        self.dist_field[self.dist_field <= self.epsilon] = (1 / (2 * self.epsilon)) * (
            self.dist_field[self.dist_field <= self.epsilon] - self.epsilon
        ) ** 2
        self.dist_field[self.dist_field > self.epsilon] = 0
        self.dist_field = torch.from_numpy(self.dist_field).type(self.dtype)

    def collisions(self, poses):
        """
        Arguments:
            poses (K * T, 3 tensor)
        Returns:
            (K * T, tensor) 1 if collision, 0 otherwise
        """
        assert poses.size() == (self.K * self.T, 3)

        utils.map.world2map(self.map, poses, out=self.scaled)

        xs = self.scaled[:, 0].long()
        ys = self.scaled[:, 1].long()

        self.perm.zero_()
        self.perm |= self.perm_reg[ys, xs]
        self.perm |= self.perm_reg[ys + self.car_padding, xs]
        self.perm |= self.perm_reg[ys - self.car_padding, xs]
        self.perm |= self.perm_reg[ys, xs + self.car_padding]
        self.perm |= self.perm_reg[ys, xs - self.car_padding]

        return self.perm.type(self.dtype)

    def check_collision_in_map(self, poses):
        assert poses.size() == (self.K * self.T, 3)

        utils.map.world2map(self.map, poses, out=self.scaled)

        L = self.car_length
        W = self.car_width

        # Specify specs of bounding box
        bbox = self.dtype(
            [
                [L / 2.0, W / 2.0],
                [L / 2.0, -W / 2.0],
                [-L / 2.0, W / 2.0],
                [-L / 2.0, -W / 2.0],
            ]
        )

        bbox.div_(self.map.resolution)

        x = bbox[:, 0].expand(len(poses), -1)
        y = bbox[:, 1].expand(len(poses), -1)

        xs = self.scaled[:, 0]
        ys = self.scaled[:, 1]
        thetas = self.scaled[:, 2]

        c = torch.cos(thetas).resize_(len(thetas), 1)
        s = torch.sin(thetas).resize_(len(thetas), 1)

        self.bbox_map[:, 0] = (x * c - y * s) + xs.unsqueeze(-1).expand(-1, 4)
        self.bbox_map[:, 1] = (x * s + y * c) + ys.unsqueeze(-1).expand(-1, 4)

        bbox_idx = self.bbox_map.long()

        self.perm.zero_()
        self.perm |= self.perm_reg[bbox_idx[:, 1, 0], bbox_idx[:, 0, 0]]
        self.perm |= self.perm_reg[bbox_idx[:, 1, 1], bbox_idx[:, 0, 1]]
        self.perm |= self.perm_reg[bbox_idx[:, 1, 2], bbox_idx[:, 0, 2]]
        self.perm |= self.perm_reg[bbox_idx[:, 1, 3], bbox_idx[:, 0, 3]]

        return self.perm.type(self.dtype)

    def distances(self, poses):
        """
        Arguments:
            poses (K * T, 3 tensor)
        Returns:
            (K * T, tensor) with distances in terms of map frame
        """

        utils.map.world2map(self.map, poses, out=self.scaled)

        xs = self.scaled[:, 0].long()
        ys = self.scaled[:, 1].long()

        return self.dist_field[ys, xs]
