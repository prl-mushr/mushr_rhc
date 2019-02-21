#!/usr/bin/env python

import rhctensor
import rospy
import threading
import rhcbase
import utils

from geometry_msgs.msg import PoseArray, PoseStamped
from visualization_msgs.msg import Marker


class Debugger(rhcbase.RHCBase):
    def __init__(self, dtype, params, logger, name):
        rospy.init_node(name, anonymous=True)

        super(rhcbase.RHCBase, self).__init__(dtype, params, logger)

        rospy.Subscriber("/rhcontroller/rollouts", PoseArray, self.cb_traj_chosen, queue_size=10)
        self.traj_chosen_pub = rospy.Publisher("~traj_chosen", Marker, queue_size=10)
        self.traj_chosen = None
        self.traj_chosen_id = 1

        ip_topic = rospy.get_param("~ip_topic")
        rospy.Subscriber(ip_topic, PoseStamped, self.cb_inferred_pose, queue_size=10)
        self.inferred_pose = None

        self.current_path_pub = rospy.Publisher("~current_path", Marker, queue_size=10)
        self.current_path = Marker()
        self.current_path.frame_id = "map"
        self.current_path.type = self.current_path.LINE_STRIP
        self.current_path.action = self.current_path.ADD
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

        self.rhcctrl = self.load_controller()

        rospy.Subscriber("/move_base_simple/goal",
                         PoseStamped, self.cb_goal, queue_size=1)

    def cb_goal(self, msg):
        goal = self.dtype(utils.rospose_to_posetup(msg.pose))
        if self.rhcctrl is not None:
            if not self.rhctrl.set_goal(goal):
                self.logger.err("That goal is unreachable, please choose another")
            else:
                self.logger.info("Goal set")

    def cb_inferred_pose(self, msg):
        self.current_path.header.stamp = rospy.Time.now()
        self.current_path.points.append(msg.pose.position)
        self.current_path_pub.publish(self.current_path)

    def cb_traj_chosen(self, msg):
        self.traj_chosen = msg.poses

    def viz_chose_traj(self):
        rate = rospy.Rate(rospy.get_param("~update_trajectory_rate"))
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
                m.color.b = 1.0
                m.scale.x = 0.03

                m.points = map(lambda x: x.position, self.traj_chosen)

                self.traj_chosen_pub.publish(m)
                self.traj_chosen_id += 1
            rate.sleep()

    def viz_cost_fn(self):
        pass

    def start(self):
        if rospy.get_param("~viz_chosen_traj", True):
            traj_chosen_t = threading.Thread(target=self.viz_chosen_traj)
            traj_chosen_t.start()
        if rospy.get_param("viz_cost_fn", False):
            cost_fn_t = threading.Thread(target=self.viz_cost_fn)
            cost_fn_t.start()
        rospy.spin()


if __name__ == '__main__':
    node = Debugger(rhctensor.float_tensor(), "rhcontroller")
    node.start()
