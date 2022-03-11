#!/usr/bin/env python

import controlnode
import threading
import signal

if __name__ == '__main__':
    node = controlnode.ControlNode("controller")

    signal.signal(signal.SIGINT, node.shutdown)

    controller = threading.Tread(start=node.start)
    controller.start()

    while controller.run:
        signal.pause()

    controller.join()
