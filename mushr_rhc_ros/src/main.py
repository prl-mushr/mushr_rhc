#!/usr/bin/env python

import rhcnode
import rhctensor
import torch

if __name__ == '__main__':
    print("staring")
    node = rhcnode.RHCNode(rhctorch.float_tensor())
    node.start("rhcontroller")
