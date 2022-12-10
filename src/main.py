#!/usr/bin/env python3

import controlnode
import threading
import signal
import time

if __name__ == '__main__':
    time.sleep(5)
    node = controlnode.ControlNode("controller")

    signal.signal(signal.SIGINT, node.shutdown)

    controller = threading.Tread(start=node.start)
    controller.start()

    while controller.run:
        signal.pause()

    controller.join()
