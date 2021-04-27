# Copyright (c) 2019, The Personal Robotics Lab, The MuSHR Team, The Contributors of MuSHR
# License: BSD 3-Clause. See LICENSE.md file in root directory.

import os

import torch


def byte_tensor():
    return torch.cuda.ByteTensor if _use_cuda() else torch.ByteTensor


def float_tensor():
    return torch.cuda.FloatTensor if _use_cuda() else torch.FloatTensor


def _use_cuda():
    return int(os.getenv("RHC_USE_CUDA", 0)) == 1 and torch.cuda.is_available()
