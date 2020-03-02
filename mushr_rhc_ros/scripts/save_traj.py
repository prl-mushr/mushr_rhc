#!/usr/bin/env python
from mushr_rhc_ros.msg import SimpleTrajectory
import trajgen
import tf.transformations
import pickle

import numpy as np
import rospy
from geometry_msgs.msg import Quaternion


def a2q(a):
    return Quaternion(*tf.transformations.quaternion_from_euler(0, 0, a))


def config2simpletraj(config):
    traj = SimpleTrajectory()
    # car pose and block pose.
    x, y, t = config[0]
    traj.block_pose.position.x = x
    traj.block_pose.position.y = y
    traj.block_pose.orientation = a2q(t)

    s, c = np.sin(t), np.cos(t)
    traj.car_pose.position.x = (-0.28 * c) + x
    traj.car_pose.position.y = (-0.28 * s) + y
    traj.car_pose.orientation = a2q(t)

    for c in config:
        traj.xs.append(c[0])
        traj.ys.append(c[1])
        traj.thetas.append(c[2])

    return traj


pathlen = 6.0
# RADS = [2.5, 4.0, 10.0]
TURN_RADS = [10.0, 20.0]
KINK_RADS = [2.0, 5.0]
turns = [("left-turn", trajgen.left_turn), ("right-turn", trajgen.right_turn)]
kinks = [("left-kink", trajgen.left_kink), ("right-kink", trajgen.right_kink)]

if __name__ == "__main__":
    rospy.init_node("controller_runner")

    config = trajgen.straight_line(pathlen)
    traj = config2simpletraj(config)
    with open("trajs/straight.pickle", "wb") as f:
        pickle.dump(traj, f)

    for r in TURN_RADS:
        for fname, f in turns:
            config = f(r, pathlen=pathlen)
            traj = config2simpletraj(config)

            with open("trajs/%s-%s.pickle" % (fname, r), "wb") as f:
                pickle.dump(traj, f)

    for r in KINK_RADS:
        for fname, f in kinks:
            config = f(r, pathlen=pathlen)
            traj = config2simpletraj(config)

            with open("trajs/%s-%s.pickle" % (fname, r), "wb") as f:
                pickle.dump(traj, f)
