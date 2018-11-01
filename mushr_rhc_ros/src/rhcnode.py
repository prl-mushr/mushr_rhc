import rospy
import torch

from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from nav_msgs.srv import GetMap
from std_msgs.msg import Empty

import logger
import parameters
import utils

import librhc
import librhc.types as types
import librhc.cost as cost
import librhc.model as model
import librhc.trajgen as trajgen
import librhc.worldrep as worldrep

motion_models = {
    "kinematic": model.Kinematics,
}

trajgens = {
    "tl": trajgen.TL,
}

cost_functions = {
    "waypoints": cost.Waypoints,
}

world_reps = {
    "simple": worldrep.Simple,
}

class RHCNode:
    def __init__(self, dtype):
        self.dtype = dtype

        self.params = parameters.RosParams()
        self.logger = logger.RosLog()

    def start(self, name):
        rospy.init_node(name, anonymous=True)  # Initialize the node

        self.load_controller()
        self.setup_pub_sub()

        rate = rospy.Rate(25)
        self.inferred_pose = None

        while not rospy.is_shutdown():
            if self.inferred_pose is not None:
                 next_ctrl = self.rhctrl.step(self.inferred_pose)
                 self.publish_ctrl(next_ctrl)
            else:
                 self.logger.warn("no inferred pose")
            rate.sleep()

    def load_controller(self):
        m = self.get_model()
        cg = self.get_ctrl_gen()
        cf = self.get_cost_fn()

        self.rhctrl = librhc.MPC(self.params, self.logger,
                            self.dtype, m, cg, cf)

    def setup_pub_sub(self):
        rospy.Subscriber("/rhc/reset", Empty, self.cb_reset, queue_size=1)
        rospy.Subscriber("/pf/pose/odom", Odometry, self.cb_odom, queue_size=10)
        rospy.Subscriber("/pp/path_goal", PoseStamped, self.cb_goal, queue_size=1)
        rospy.Subscriber("/sim_car_pose/pose", PoseStamped, self.cb_pose, queue_size=10)

        self.rp_ctrls = rospy.Publisher(
            self.params.get_str(
                "~ctrl_topic",
                default="/vesc/high_level/ackermann_cmd_mux/input/nav_0"
            ),
            AckermannDriveStamped, queue_size=2
        )

    def cb_reset(self, msg):
        self.load_controller()

    def cb_odom(self, msg):
        self.inferred_pose = self.dtype(utils.pose_to_config(msg.pose.pose))

    def cb_goal(self, msg):
        goal = self.dtype([
            msg.pose.position.x,
            msg.pose.position.y,
            utils.rosquaternion_to_angle(msg.pose.orientation)
        ])
        self.rhctrl.model.set_goal(goal)

    def cb_pose(self, msg):
        self.inferred_pose = self.dtype([
            msg.pose.position.x,
            msg.pose.position.y,
            utils.rosquaternion_to_angle(msg.pose.orientation)
        ])

    def publish_ctrl(self, ctrl):
        ctrlmsg = AckermannDriveStamped()
        ctrlmsg.header.stamp = rospy.Time.now()
        ctrlmsg.header.seq = self.ackermann_msg_id
        ctrlmsg.drive.steering_angle = ctrl[0]
        ctrlmsg.drive.speed = ctrl[1]
        self.rp_ctrls.publish(ctrlmsg)
        self.ackermann_msg_id += 1

    def get_model(self):
        mname = self.params.get_str("model", default="kinematic")
        if mname not in motion_models:
            self.logger.fatal("model '{}' is not valid".format(mname))
        return motion_models[mname](self.params, self.logger, self.dtype)

    def get_ctrl_gen(self):
        tgname = self.params.get_str("ctrl_generator", default="tl")
        if tgname not in trajgens:
            self.logger.fatal("ctrl_gen '{}' is not valid".format(tgname))
        return trajgens[tgname](self.params, self.logger, self.dtype)

    def get_map(self):
        srv_name = self.params.get_str("static_map", default="/static_map")
        rospy.wait_for_service(srv_name)

        map_msg = rospy.ServiceProxy(srv_name, GetMap)().map
	x, y, angle = utils.rospose_to_posetup(map_msg.info.origin)

        return types.MapData(
            resolution = map_msg.info.resolution,
            origin_x = x,
            origin_y = y,
            orientation_angle = angle,
            width = map_msg.info.width,
            height = map_msg.info.height,
            data = map_msg.data
        )

    def get_cost_fn(self):
        cfname = self.params.get_str("cost_fn", default="waypoints")
        if cfname not in cost_functions:
            self.logger.fatal("cost_fn '{}' is not valid".format(cfname))

        wrname = self.params.get_str("world_rep", default="simple")
        if wrname not in world_reps:
            self.logger.fatal("world_rep '{}' is not valid".format(wrname))

        wr = world_reps[wrname](self.params, self.logger, self.dtype, self.get_map())

        return cost_functions[cfname](self.params, self.logger, self.dtype, wr)
