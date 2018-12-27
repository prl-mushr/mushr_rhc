#!/usr/bin/env python

import rhcnode
import rhctensor

if __name__ == '__main__':
    print("Starting RHController Node")
    node = rhcnode.RHCNode(rhctensor.float_tensor())
    node.start("rhcontroller")
