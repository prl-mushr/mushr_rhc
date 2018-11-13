import numpy as np
import os
import torch

from scipy import signal

class Simple:
    def __init__(self, params, logger, dtype, map):
        """
            Inputs:
                params (obj): parameter getter
                logger (obj): a logger to write to
                dtype (obj): data type for tensors
                map (obj): A map representation
        """
        self.T = params.get_int("T", default=15)
        self.K = params.get_int("K", default=62)
        self.params = params
        self.logger = logger
        self.dtype = dtype
        self.map = map

        # Ratio of car to extend in every direction
        # TODO: project car into its actual orientation
        self.car_ratio = params.get_float("world_rep/car_ratio", default=3.2) # was 3.2
        self.car_length = params.get_float("world_rep/car_length", default=0.37)
        self.car_padding = long((self.car_length / self.map.resolution) / self.car_ratio)
        self._load_permissible_region()

        self.scaled = self.dtype(self.K * self.T, 3)
        self.perm = torch.cuda.ByteTensor(self.K * self.T)

    def collisions(self, poses):
        """
        Arguments:
            poses (K * T, 3 tensor)
        """
        assert poses.size() == (self.K * self.T, 3)

        self._world_to_map(poses)

        xs = self.scaled[:, 0].long()
        ys = self.scaled[:, 1].long()

        self.perm.zero_()
        self.perm |= self.perm_reg[ys, xs]
        self.perm |= self.perm_reg[ys + self.car_padding, xs]
        self.perm |= self.perm_reg[ys - self.car_padding, xs]
        self.perm |= self.perm_reg[ys, xs + self.car_padding]
        self.perm |= self.perm_reg[ys, xs - self.car_padding]

        for i in range(self.K * self.T):
            print "x {} y {} collision {}".format(xs[i], ys[i], self.perm[i])

        return self.perm.type(self.dtype)

    def _world_to_map(self, poses):
        # equivalent to map_to_grid(world_to_map(poses))
        # operates in place
        scale = float(self.map.resolution)

        self.scaled.copy_(poses)

        # translation
        self.scaled[:, 0].sub_(self.map.origin_x).mul_(1.0/scale)
        self.scaled[:, 1].sub_(self.map.origin_y).mul_(1.0/scale)
        self.scaled[:, 2] += self.map.angle

        xs = self.scaled[:, 0]
        ys = self.scaled[:, 1]

        # we need to store the x coordinates since they will be overwritten
        xs_p = xs.clone()

        xs = xs * self.map.angle_cos - ys * self.map.angle_sin
        ys = xs_p * self.map.angle_sin + ys * self.map.angle_cos

    def _load_permissible_region(self):
        # perm_reg_file = '/media/JetsonSSD/permissible_region/' + map_name
        path = self.params.get_str(
            'world_rep/permissible_region_dir',
             default='/tmp/permissible_region/'
        )
        name = self.params.get_str('world_rep/map_name', default="foo")
        perm_reg_file = path + name

        if os.path.isfile(perm_reg_file + '.npy'):
            pr = np.load(perm_reg_file + '.npy')
        else:
            array_255 = np.array(self.map.data).reshape((self.map.height, self.map.width))
            pr = np.zeros_like(array_255, dtype=bool)

            # Numpy array of dimension (map_msg.info.height, map_msg.info.width),
            # With values 0: not permissible, 1: permissible
            pr[array_255 == 0] = 1
            pr = np.logical_not(pr) # 0 is permissible, 1 is not

            KERNEL_SIZE = 31 # 15 cm = 7 pixels = kernel size 15x15
            kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE))
            kernel /= kernel.sum()
            pr = signal.convolve2d(pr, kernel, mode='same') > 0 # boolean 2d array
            np.save(perm_reg_file, pr)

        self.perm_reg = torch.from_numpy(pr.astype(np.int)).type(torch.cuda.ByteTensor)
