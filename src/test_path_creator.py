#!/usr/bin/env python3

import rospy
from mushr_rhc.msg import XYHVPath, XYHV
import numpy as np

rospy.init_node("traj_pub", anonymous=True)

traj_pub = rospy.Publisher("controller/set_path", XYHVPath, queue_size=2)

trajectory = XYHVPath()
for i in range(10):
    point = XYHV()
    point.x = i
    point.y = i
    point.v = 1
    point.h = 45/57.3
    trajectory.waypoints.append(point)

point = XYHV()
point.x = 10
point.y = 10
point.v = 0
point.h = 45/57.3

trajectory.waypoints.append(point)

r = rospy.Rate(1)
traj_pub.publish(trajectory)
r.sleep()
traj_pub.publish(trajectory)

