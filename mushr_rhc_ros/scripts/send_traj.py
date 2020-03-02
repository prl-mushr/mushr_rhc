#!/usr/bin/env python
from mushr_rhc_ros.msg import SimpleTrajectory
import time
import tf.transformations
import trajgen

import numpy as np
import rospy
from geometry_msgs.msg import Quaternion


def a2q(a):
    return Quaternion(*tf.transformations.quaternion_from_euler(0, 0, a))


plans = {
    'circle': trajgen.circle,
    'wave': trajgen.wave,
    'striaght_line': trajgen.straight_line,
    'left-10': lambda: trajgen.left_turn(10.0, 6.0),
    'left-4': lambda: trajgen.left_turn(4.0, 6.0),
    'left-2.5': lambda: trajgen.left_turn(2.5, 6.0),
    'right-10': lambda: trajgen.right_turn(10.0, 6.0),
    'right-4': lambda: trajgen.right_turn(4.0, 6.0),
    'right-2.5': lambda: trajgen.right_turn(2.5, 6.0),
    'left-kink-3.0': lambda: trajgen.left_kink(3.0, 6.),
    'left-kink-5.0': lambda: trajgen.left_kink(5.0, 6.),
    'right-kink-2.0': lambda: trajgen.right_kink(2.0, 6.),
    'right-kink-5.0': lambda: trajgen.right_kink(5.0, 6.),
    'real': trajgen.real_traj,
}


def get_plan():
    print "Which plan would you like to generate? "
    plan_names = plans.keys()
    for i, name in enumerate(plan_names):
        print "{} ({})".format(name, i)
    index = int(raw_input("num: "))
    if index >= len(plan_names):
        print "Wrong number. Exiting."
        exit()
    return plans[plan_names[index]]()


if __name__ == "__main__":
    rospy.init_node("controller_runner")
    configs = get_plan()

    # h = Header()
    # h.stamp = rospy.Time.now()

    # desired_speed = 2.0
    # ramp_percent = 0.1
    # ramp_up = np.linspace(0.0, desired_speed, int(ramp_percent * len(configs)))
    # ramp_down = np.linspace(desired_speed, 0.3, int(ramp_percent * len(configs)))
    # speeds = np.zeros(len(configs))
    # speeds[:] = desired_speed
    # speeds[0:len(ramp_up)] = ramp_up
    # speeds[-len(ramp_down):] = ramp_down

    traj = SimpleTrajectory()
    # car pose and block pose.
    x, y, t = configs[0]
    traj.block_pose.position.x = x
    traj.block_pose.position.y = y
    traj.block_pose.orientation = a2q(t)

    s, c = np.sin(t), np.cos(t)
    traj.car_pose.position.x = (-0.28 * c) + x
    traj.car_pose.position.y = (-0.28 * s) + y
    traj.car_pose.orientation = a2q(t)

    for c in configs:
        traj.xs.append(c[0])
        traj.ys.append(c[1])
        traj.thetas.append(c[2])

    p = rospy.Publisher("/rhcontroller/trajectory", SimpleTrajectory, queue_size=1)
    time.sleep(1)

    print "Sending path..."
    p.publish(traj)
    print "Controller started."
