#!/usr/bin/env python

import rhcnode
import rhctensor
import cProfile as cp
import os

if __name__ == '__main__':
    print("Starting RHController Node")
    node = rhcnode.RHCNode(rhctensor.float_tensor())
    pr = cp.Profile()
    pr.enable()
    node.start("rhcontroller")
    pr.disable()
    pr.dump_stats(os.path.expanduser("~/profile.prof"))
