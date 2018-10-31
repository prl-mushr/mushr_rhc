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

    def collisions(self, poses):
        """
        Arguments:
            poses (K * T, 3 tensor)
        """

        self.world_to_map(poses)

        xs = self.scaled[:,0].long()
        ys = self.scaled[:,1].long()

        perm = self.perm_reg[ys.long(), xs.long()] # (T*K,) with map value 0 or 1
        perm |= self.perm_reg[ys.long()+self.car_padding, xs.long()] # (T*K,) with map value 0 or 1
        perm |= self.perm_reg[ys.long()-self.car_padding, xs.long()] # (T*K,) with map value 0 or 1
        perm |= self.perm_reg[ys.long(), xs.long()+self.car_padding] # (T*K,) with map value 0 or 1
        perm |= self.perm_reg[ys.long(), xs.long()-self.car_padding] # (T*K,) with map value 0 or 1
        perm = perm.type(self.dtype)

	return perm

    def world_to_map(self, poses):
        # equivalent to map_to_grid(world_to_map(poses))
        # operates in place
        scale = self.map.resolution

        self.scaled.copy_(poses)
        # translation
        xs = self.scaled[:, 0]
        ys = self.scaled[:, 1]
        xs -= self.map.origin_x
        ys -= self.map.origin_y

        # scale
        self.scaled[:, :2] *= (1.0 / float(scale))

        # we need to store the x coordinates since they will be overwritten
        xs_p = xs.clone()

        xs *= self.map.angle_cos
        xs -= ys * self.map.angle_sin
        ys *= self.map.angle_cos
        ys += xs_p * self.map.angle_sin
        self.scaled[:, 2] += self.map.angle

    def _load_permissible_region(self):
	# perm_reg_file = '/media/JetsonSSD/permissible_region/' + map_name
        path = self.params.get_str(
            'world_rep/map_file_location',
             default='/media/JetsonSSD/permissible_region/'
        )
        name = self.params.get_str('world_rep/map_name', default="identical_rooms")
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

	self.perm_reg = torch.from_numpy(pr.astype(np.int)).type(self.dtype)
