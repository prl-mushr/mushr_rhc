#!/usr/bin/env python

import torch
import rhcnode

FLOAT_TENSOR = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

if __name__ == '__main__':
    print("staring")
    node = rhcnode.RHCNode(FLOAT_TENSOR)
    node.start("rhcontroller")
