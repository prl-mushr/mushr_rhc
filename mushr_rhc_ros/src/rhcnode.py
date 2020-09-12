#!/usr/bin/env python

# Copyright (c) 2019, The Personal Robotics Lab, The MuSHR Team, The Contributors of MuSHR
# License: BSD 3-Clause. See LICENSE.md file in root directory.

import cProfile
import os
import signal
import threading

import rospy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Point, PoseStamped
from std_msgs.msg import ColorRGBA, Empty
from std_srvs.srv import Empty as SrvEmpty
from visualization_msgs.msg import Marker

import logger
import parameters
import rhcbase
import rhctensor
import utils


class RHCNode(rhcbase.RHCBase):
    def __init__(self, dtype, params, logger, name):
        rospy.init_node(name, anonymous=True, disable_signals=True)

        super(RHCNode, self).__init__(dtype, params, logger)

        self.reset_lock = threading.Lock()
        self.inferred_pose_lock = threading.Lock()
        self._inferred_pose = None

        self.cur_rollout = self.cur_rollout_ip = None
        self.traj_pub_lock = threading.Lock()

        self.goal_event = threading.Event()
        self.map_metadata_event = threading.Event()
        self.ready_event = threading.Event()
        self.events = [self.goal_event, self.map_metadata_event, self.ready_event]
        self.run = True

        self.do_profile = self.params.get_bool("profile", default=False)

    def start_profile(self):
        if self.do_profile:
            self.logger.warn("Running with profiling")
            self.pr = cProfile.Profile()
            self.pr.enable()

    def end_profile(self):
        if self.do_profile:
            self.pr.disable()
            self.pr.dump_stats(os.path.expanduser("~/mushr_rhc_stats.prof"))

    def start(self):
        self.logger.info("Starting RHController")
        self.start_profile()
        self.setup_pub_sub()
        self.rhctrl = self.load_controller()
        self.T = self.params.get_int("T")

        self.ready_event.set()

        rate = rospy.Rate(50)
        self.logger.info("Initialized")

        while not rospy.is_shutdown() and self.run:
            ip = self.inferred_pose()
            next_traj, rollout = self.run_loop(ip)
            with self.traj_pub_lock:
                if rollout is not None:
                    self.cur_rollout = rollout.clone()
                    self.cur_rollout_ip = ip

            if next_traj is not None:
                self.publish_traj(next_traj, rollout)
                # For experiments. If the car is at the goal, notify the
                # experiment tool
                if self.rhctrl.at_goal(self.inferred_pose()):
                    self.expr_at_goal.publish(Empty())
                    self.goal_event.clear()
            rate.sleep()

        self.end_profile()

    def run_loop(self, ip):
        self.goal_event.wait()
        if rospy.is_shutdown() or ip is None:
            return None, None
        with self.reset_lock:
            # If a reset is initialed after the goal_event was set, the goal
            # will be cleared. So we have to have another goal check here.
            if not self.goal_event.is_set():
                return None, None
            if ip is not None:
                return self.rhctrl.step(ip)
            self.logger.err("Shouldn't get here: run_loop")

    def shutdown(self, signum, frame):
        rospy.signal_shutdown("SIGINT recieved")
        self.run = False
        for ev in self.events:
            ev.set()

    def setup_pub_sub(self):
        rospy.Service("~reset/soft", SrvEmpty, self.srv_reset_soft)
        rospy.Service("~reset/hard", SrvEmpty, self.srv_reset_hard)

        car_name = self.params.get_str("car_name", default="car")

        rospy.Subscriber(
            "/move_base_simple/goal", PoseStamped, self.cb_goal, queue_size=1
        )

        rospy.Subscriber(
            "/" + car_name + "/" + rospy.get_param("~inferred_pose_t"),
            PoseStamped,
            self.cb_pose,
            queue_size=10,
        )

        self.rp_ctrls = rospy.Publisher(
            "/"
            + car_name
            + "/"
            + self.params.get_str(
                "ctrl_topic", default="mux/ackermann_cmd_mux/input/navigation"
            ),
            AckermannDriveStamped,
            queue_size=2,
        )

        traj_chosen_t = self.params.get_str("traj_chosen_topic", default="~traj_chosen")
        self.traj_chosen_pub = rospy.Publisher(traj_chosen_t, Marker, queue_size=10)

        # For the experiment framework, need indicators to listen on
        self.expr_at_goal = rospy.Publisher("experiments/finished", Empty, queue_size=1)

    def srv_reset_hard(self, msg):
        """
        Hard reset does a complete reload of the controller
        """
        rospy.loginfo("Start hard reset")
        self.reset_lock.acquire()
        self.load_controller()
        self.goal_event.clear()
        self.reset_lock.release()
        rospy.loginfo("End hard reset")
        return []

    def srv_reset_soft(self, msg):
        """
        Soft reset only resets soft state (like tensors). No dependencies or maps
        are reloaded
        """
        rospy.loginfo("Start soft reset")
        self.reset_lock.acquire()
        self.rhctrl.reset()
        self.goal_event.clear()
        self.reset_lock.release()
        rospy.loginfo("End soft reset")
        return []

    def cb_goal(self, msg):
        goal = self.dtype(utils.rospose_to_posetup(msg.pose))
        self.ready_event.wait()
        if not self.rhctrl.set_goal(goal):
            self.logger.err("That goal is unreachable, please choose another")
            return
        else:
            self.logger.info("Goal set")
        self.goal_event.set()

    def cb_pose(self, msg):
        self.set_inferred_pose(self.dtype(utils.rospose_to_posetup(msg.pose)))

        if self.cur_rollout is not None and self.cur_rollout_ip is not None:
            m = Marker()
            m.header.frame_id = "map"
            m.type = m.LINE_STRIP
            m.action = m.ADD
            with self.traj_pub_lock:
                pts = (
                    self.cur_rollout[:, :2] - self.cur_rollout_ip[:2]
                ) + self.inferred_pose()[:2]

            m.points = map(lambda xy: Point(x=xy[0], y=xy[1]), pts)

            r, g, b = 0x36, 0xCD, 0xC4
            m.colors = [ColorRGBA(r=r / 255.0, g=g / 255.0, b=b / 255.0, a=0.7)] * len(
                m.points
            )
            m.scale.x = 0.05
            self.traj_chosen_pub.publish(m)

    def publish_traj(self, traj, rollout):
        assert traj.size() == (self.T, 2)
        assert rollout.size() == (self.T, 3)

        ctrl = traj[0]
        ctrlmsg = AckermannDriveStamped()
        ctrlmsg.header.stamp = rospy.Time.now()
        ctrlmsg.drive.speed = ctrl[0]
        ctrlmsg.drive.steering_angle = ctrl[1]
        self.rp_ctrls.publish(ctrlmsg)

    def set_inferred_pose(self, ip):
        with self.inferred_pose_lock:
            self._inferred_pose = ip

    def inferred_pose(self):
        with self.inferred_pose_lock:
            return self._inferred_pose


if __name__ == "__main__":
    params = parameters.RosParams()
    logger = logger.RosLog()
    node = RHCNode(rhctensor.float_tensor(), params, logger, "rhcontroller")

    signal.signal(signal.SIGINT, node.shutdown)
    rhc = threading.Thread(target=node.start)
    rhc.start()

    # wait for a signal to shutdown
    while node.run:
        signal.pause()

    rhc.join()
