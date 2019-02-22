#!/usr/bin/env python

import logger
import parameters
import rhcbase
import rhctensor
import rospy
import threading
import utils
import cProfile
import os

from geometry_msgs.msg import PoseArray, PoseStamped, PoseWithCovarianceStamped
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
        self.current_path.color.b = 1.0
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

    def start_profile(self):
        if self.do_profile:
            self.logger.warn("Running with profiling")
            self.pr = cProfile.Profile()
            self.pr.enable()

    def end_profile(self):
        if self.do_profile:
            self.pr.disable()
            self.pr.dump_stats(os.path.expanduser('~/mushr_rhc_stats.prof'))

    def cb_goal(self, msg):
        goal = self.dtype(utils.rospose_to_posetup(msg.pose))
        self.logger.info("Got goal")
        if self.rhctrl is not None:
            self.start_profile()
            if not self.rhctrl.set_goal(goal):
                self.logger.err("That goal is unreachable, please choose another")
            else:
                self.logger.info("Goal set")
                self.goal = goal
            self.end_profile()

    def cb_initialpose(self, msg):
        self.init_pose = self.dtype(utils.rospose_to_posetup(msg.pose.pose))

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
            if self.goal is not None:
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
