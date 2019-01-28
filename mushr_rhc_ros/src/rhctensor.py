import os
import torch


def byte_tensor():
    return torch.cuda.ByteTensor if _use_cuda() else torch.ByteTensor


def float_tensor():
    return torch.cuda.FloatTensor if _use_cuda() else torch.FloatTensor


def _use_cuda():
    return int(os.getenv("RHC_USE_CUDA", 0)) == 1 and torch.cuda.is_available()
