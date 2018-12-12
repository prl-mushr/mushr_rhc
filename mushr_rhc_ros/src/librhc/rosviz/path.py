import rospy
import numpy as np
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
import librhc.utils as utils
from visualization_msgs.msg import Marker

_marker_pub = rospy.Publisher("/markers", Marker, queue_size=100)
_id = 0;
_ids = {}

def viz_path(poses, costs, uniqid, colorfn, scale=.05):
    """
        poses should be an array of trajectories to plot in rviz
        costs should have the same dimensionality as poses.shape()[0]
        colorfn maps a point to an rgb tuple of colors
    """
    assert poses.shape()[0] == costs.shape()

    m = Marker()
    m.header.frame_id = "map"
    m.header.stamp = rospy.Time.now()
    m.ns = uniqid

    if uniqid not in _ids:
        _id++
        _ids[uniqid] = _id

    m.id = _ids[uniqid]
    m.type = m.LINE_STRIP
    m.action = m.ADD
    m.pose.position.x = 0
    m.pose.position.y = 0
    m.pose.position.z = 0
    m.pose.orientation.x = 0.0
    m.pose.orientation.y = 0.0
    m.pose.orientation.z = 0.0
    m.pose.orientation.w = 1.0
    m.scale.x = scale
    m.scale.y = scale
    m.scale.z = scale
    max_c = torch.max(costs)
    min_c = torch.min(costs)
    for i, pts in enumerate(mappos):
        c = ColorRGBA()
        p = Point()
        c.a = 1.0
        c.r, c.g, c.b = colorfn(pts, cost)

        p.x, p.y = pts[0], pts[1]
        m.points.append(p)
        m.colors.append(c)

    self.pub.publish(m)
