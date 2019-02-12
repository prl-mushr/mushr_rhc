#!/usr/bin/env python

import rhcnode
import rhctensor
import threading
import signal


if __name__ == '__main__':
    node = rhcnode.RHCNode(rhctensor.float_tensor(), "rhcontroller")

    signal.signal(signal.SIGINT, node.shutdown)
    rhc = threading.Thread(target=node.start)
    rhc.start()

    # wait for a signal to shutdown
    while node.run:
        signal.pause()

    rhc.join()
