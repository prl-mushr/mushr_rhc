import os
import torch

import rhctensor

import librhc.utils as utils

class Simple:
    def __init__(self, params, logger, dtype, map):
        """
            Inputs:
                params (obj): parameter getter
                logger (obj): a logger to write to
                dtype (obj): data type for tensors
                map (types.MapData): A map representation
        """
        self.T = params.get_int("T", default=15)
        self.K = params.get_int("K", default=62)
        self.params = params
        self.logger = logger
        self.dtype = dtype
        self.map = map

        # Ratio of car to extend in every direction
        # TODO: project car into its actual orientation
        self.car_ratio = params.get_float("world_rep/car_ratio", default=1.0) # was 3.2
        self.car_length = params.get_float("world_rep/car_length", default=0.33)
        self.car_padding = long((self.car_length / self.map.resolution) / self.car_ratio)

        self.perm_reg = utils.load_permissible_region(self.params, self.map)

        self.scaled = self.dtype(self.K, 3)
        self.perm = rhctensor.byte_tensor()(self.K)

    def collisions(self, poses):
        """
        Arguments:
            poses (K * T, 3 tensor)
        Returns:
            (K * T, tensor) 1 if collision, 0 otherwise
        """
        #assert poses.size() == (self.K * self.T, 3)

        utils.world2map(self.map, poses, out=self.scaled)

        xs = self.scaled[:, 0].long()
        ys = self.scaled[:, 1].long()

        self.perm.zero_()
        self.perm |= self.perm_reg[ys, xs]
        self.perm |= self.perm_reg[ys + self.car_padding, xs]
        self.perm |= self.perm_reg[ys - self.car_padding, xs]
        self.perm |= self.perm_reg[ys, xs + self.car_padding]
        self.perm |= self.perm_reg[ys, xs - self.car_padding]

        return self.perm.type(self.dtype)
