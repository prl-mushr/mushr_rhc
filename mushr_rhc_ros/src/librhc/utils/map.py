from scipy import signal

import numpy as np
import os
import rhctensor
import torch

def world2map(mapdata, poses, out=None):
    if out is None:
        panic("out cannot be None")

    assert poses.size() == out.size()

    out[:,:] = poses
    scale = float(mapdata.resolution)

    # translation
    out[:, 0].sub_(mapdata.origin_x).mul_(1.0/scale)
    out[:, 1].sub_(mapdata.origin_y).mul_(1.0/scale)
    out[:, 2] += mapdata.angle

    xs = out[:, 0]
    ys = out[:, 1]

    # we need to store the x coordinates since they will be overwritten
    xs_p = xs.clone()

    out[:,0] = xs   * mapdata.angle_cos - ys * mapdata.angle_sin
    out[:,1] = xs_p * mapdata.angle_sin + ys * mapdata.angle_cos

def world2mapnp(mapdata, poses):
    # translation
    poses[:, 0] -= mapdata.origin_x
    poses[:, 1] -= mapdata.origin_y

    # scale
    poses[:, :2] *= (1.0 / float(mapdata.resolution))

    # we need to store the x coordinates since they will be overwritten
    temp = np.copy(poses[:, 0])
    poses[:, 0] = mapdata.angle_cos * poses[:, 0] - mapdata.angle_sin * poses[:, 1]
    poses[:, 1] = mapdata.angle_sin * temp + mapdata.angle_cos * poses[:, 1]
    poses[:, 2] += mapdata.angle

def map2worldnp(mapdata, poses):
    # rotation
    # we need to store the x coordinates since they will be overwritten
    temp = np.copy(poses[:, 0])
    poses[:, 0] = mapdata.angle_cos * poses[:, 0] - mapdata.angle_sin * poses[:, 1]
    poses[:, 1] = mapdata.angle_sin * temp + mapdata.angle_cos * poses[:, 1]

    # scale
    poses[:, :2] *= float(mapdata.resolution)

    # translate
    poses[:, 0] += mapdata.origin_x
    poses[:, 1] += mapdata.origin_y
    poses[:, 2] += mapdata.angle

def load_permissible_region(params, map):
    path = params.get_str(
        'permissible_region_dir',
         default='/media/JetsonSSD/permissible_region/'
    )
    name = params.get_str('map_name', default="default_map")

    perm_reg_file = path + name

    if not os.path.isdir(path):
        print "Directory " + path + " doesn't exist"
        exit(1)

    print "loading map"
    if os.path.isfile(perm_reg_file + '.npy'):
        pr = np.load(perm_reg_file + '.npy')
    else:
        array_255 = np.array(map.data).reshape((map.height, map.width))
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

    return torch.from_numpy(pr.astype(np.int)).type(rhctensor.byte_tensor())
