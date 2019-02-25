#!/usr/bin/env python

import cProfile
import logger
import matplotlib.cm as cm
import matplotlib.colors as mplcolors
import os
import parameters
import rhcbase
import rhctensor
import rospy
import threading
import torch
import utils

from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point, PoseArray, PoseStamped, PoseWithCovarianceStamped
from visualization_msgs.msg import Marker


class RHCDebug(rhcbase.RHCBase):
    def __init__(self, dtype, params, logger, name):
        rospy.init_node(name, anonymous=True, log_level=rospy.DEBUG)

        super(RHCDebug, self).__init__(dtype, params, logger)
        self.do_profile = True

        self.traj_chosen = None
        self.traj_chosen_id = 1

        self.inferred_pose = None
        self.init_pose = None

        self.goal = None
        self.debug_rollouts = self.params.get_bool("debug/flag/rollouts_on_init_pose", default=False)
        self.debug_current_path = self.params.get_bool("debug/flag/current_path", default=False)

        self.current_path = Marker()
        self.current_path.header.frame_id = "map"
        self.current_path.type = self.current_path.LINE_STRIP
        self.current_path.action = self.current_path.ADD
        self.current_path.id = 1
        self.current_path.pose.position.x = 0
        self.current_path.pose.position.y = 0
        self.current_path.pose.position.z = 0
        self.current_path.pose.orientation.x = 0.0
        self.current_path.pose.orientation.y = 0.0
        self.current_path.pose.orientation.z = 0.0
        self.current_path.pose.orientation.w = 1.0
        self.current_path.color.a = 1.0
        self.current_path.color.r = 1.0
        self.current_path.scale.x = 0.03

        self.rhctrl = self.load_controller()

        rospy.Subscriber("/initialpose", PoseWithCovarianceStamped, self.cb_initialpose)

        if self.debug_current_path:
            ip_topic = self.params.get_str("debug/ip_topic")
            rospy.Subscriber(ip_topic, PoseStamped, self.cb_inferred_pose, queue_size=10)

            self.current_path_pub = rospy.Publisher("~current_path", Marker, queue_size=10)

        traj_chosen_topic = self.params.get_str("traj_chosen_topic")
        rospy.Subscriber(traj_chosen_topic, PoseArray, self.cb_traj_chosen, queue_size=10)
        self.traj_chosen_pub = rospy.Publisher("~traj_chosen", Marker, queue_size=10)

        rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.cb_goal, queue_size=1)
        self.goal_pub = rospy.Publisher("~goal", Marker, queue_size=10)

        # self.value_heat_map_pub = rospy.Publisher("~value_fn", Marker, queue_size=100)
        # self.pub_heat_map()

    def cb_goal(self, msg):
        goal = self.dtype(utils.rospose_to_posetup(msg.pose))
        self.logger.info("Got goal")
        if self.rhctrl is not None:
            if not self.rhctrl.set_goal(goal):
                self.logger.err("That goal is unreachable, please choose another")
            else:
                self.logger.info("Goal set")
                self.goal = goal
                m = Marker()
                m.header.frame_id = "map"
                m.header.stamp = rospy.Time.now()
                m.id = 1
                m.type = m.ARROW
                m.action = m.ADD
                m.pose = msg.pose
                m.color.r = 1.0
                m.color.b = 1.0
                m.scale.x = 1
                m.scale.y = 0.1
                m.scale.z = 0.1
                self.goal_pub.publish(m)

    def cb_initialpose(self, msg):
        self.init_pose = self.dtype(utils.rospose_to_posetup(msg.pose.pose))

        self.logger.info("Got initial pose")

        if self.debug_current_path:
            # If the current path already exists, delete it.
            self.current_path.action = self.current_path.DELETE
            self.current_path_pub.publish(self.current_path)
            self.current_path.action = self.current_path.ADD

        if self.debug_rollouts:
            if self.goal is not None:
                # There is viz_logic in here, so don't do anything with the return
                self.rhctrl.step(self.init_pose)
            else:
                self.logger.info("No goal set")

    def cb_inferred_pose(self, msg):
        if self.init_pose is not None:
            self.current_path.header.stamp = rospy.Time.now()
            self.current_path.points.append(msg.pose.position)
            self.current_path_pub.publish(self.current_path)
        self.inferred_pose = self.dtype(utils.rospose_to_posetup(msg.pose))

    def cb_traj_chosen(self, msg):
        self.traj_chosen = msg.poses
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = rospy.Time.now()
        m.id = 1
        m.type = m.LINE_STRIP
        m.action = m.ADD
        m.pose.position.x = 0
        m.pose.position.y = 0
        m.pose.position.z = 0
        m.pose.orientation.x = 0.0
        m.pose.orientation.y = 0.0
        m.pose.orientation.z = 0.0
        m.pose.orientation.w = 1.0
        m.color.a = 1.0
        m.color.g = 1.0
        m.scale.x = 0.03

        m.points = map(lambda x: x.position, self.traj_chosen)

        self.traj_chosen_pub.publish(m)
        self.traj_chosen_id += 1

    def pub_heat_map(self):
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = rospy.Time.now()
        m.id = 1
        m.type = m.POINTS
        m.action = m.ADD
        m.pose.position.x = self.map_data.origin_x
        m.pose.position.y = self.map_data.origin_y
        m.pose.position.z = 0
        m.pose.orientation = utils.angle_to_rosquaternion(self.map_data.angle)
        m.color.a = 1.0
        m.color.g = 1.0
        m.scale.x = 0.5

        rospoints = []

        for i in range(150, self.map_data.width - 150, 50):
            for j in range(150, self.map_data.height - 150, 50):
                rospoints.append(self.dtype([i , j]).mul_(self.map_data.resolution))

        print self.map_data.resolution
        rospoints = torch.stack(rospoints)
        print rospoints

        print self.map_data.height, self.map_data.width
        K = self.params.get_int("K")
        T = self.params.get_int("T")

        # Filter colliding points
        collisions = self.dtype(K * T, 3)
        for i in range(0, len(rospoints), K * T):
            print i
            end = min(len(rospoints) - i, K * T)
            collisions[:end, :2] = rospoints[i:i+end]
            col = self.rhctrl.cost.world_rep.collisions(collisions)
            for p, c in zip(collisions[:end], col[:end]):
                if c == 0:
                    m.points.append(p)


        points = self.dtype(K, 3)
        colors = []
        for i in range(0, len(m.points), K):
            end = min(len(m.points) - i, K)
            points[:end, 0] = self.dtype(map(lambda p: p.x, m.points[i: i+end]))
            points[:end, 1] = self.dtype(map(lambda p: p.y, m.points[i: i+end]))

            c2g = self.rhctrl.cost.value_fn.get_value(points).mul(cost2go_w)

            colors.extend(map(float, list(c2g)[:end]))

        print colors

        norm = mplcolors.Normalize(vmin=min(colors), vmax=max(colors))
        cmap = cm.get_cmap('coolwarm')

        def colorfn(cost):
            col = cmap(norm(cost))
            r, g, b, a = col[0], col[1], col[2], 1.0
            if len(col) > 3:
                a = col[3]
            return ColorRGBA(r=r, g=g, b=b, a=a)

        m.colors = map(colorfn, colors)
        self.value_heat_map_pub.publish(m)

    def viz_traj_chosen_trace(self):
        rate = rospy.Rate(self.params.get_int("debug/traj_chosen_trace/rate"))
        while not rospy.is_shutdown():
            if self.traj_chosen is not None:
                m = Marker()
                m.header.frame_id = "map"
                m.header.stamp = rospy.Time.now()
                m.id = self.traj_chosen_id
                m.type = m.LINE_STRIP
                m.action = m.ADD
                m.pose.position.x = 0
                m.pose.position.y = 0
                m.pose.position.z = 0
                m.pose.orientation.x = 0.0
                m.pose.orientation.y = 0.0
                m.pose.orientation.z = 0.0
                m.pose.orientation.w = 1.0
                m.color.a = 1.0
                m.color.g = 1.0
                m.scale.x = 0.03

                m.points = map(lambda x: x.position, self.traj_chosen)

                self.traj_chosen_pub.publish(m)
                self.traj_chosen_id += 1
            rate.sleep()

    def viz_cost_fn(self):
        rate = rospy.Rate(100)
        while not rospy.is_shutdown():
            if self.goal is not None and self.inferred_pose is not None:
                # There is viz_logic in here, so don't do anything with the return
                self.rhctrl.step(self.inferred_pose)
            rate.sleep()

    def start(self):
        # If we are trying to debug our rollouts, we only want to run
        # the loop on initial pose. This way of implementing it could be
        # changed, but for now this will get the job done
        if self.params.get_bool("~debug/flag/viz_traj_chosen_trace", True):
            traj_chosen_trace_t = threading.Thread(target=self.viz_traj_chosen_trace)
            traj_chosen_trace_t.start()
        if self.params.get_bool("~debug/flag/viz_cost_fn", False):
            cost_fn_t = threading.Thread(target=self.viz_cost_fn)
            cost_fn_t.start()
        rospy.spin()


if __name__ == '__main__':
    params = parameters.RosParams()
    logger = logger.RosLog()
    node = RHCDebug(rhctensor.float_tensor(), params, logger, "rhcdebugger")
    node.start()
