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
import librhc.cost as cost
import librhc.model as model
import librhc.trajgen as trajgen
import librhc.types as types
import librhc.value as value
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

value_functions = {
    "simpleknn": value.SimpleKNN,
}

world_reps = {
    "simple": worldrep.Simple,
}


class RHCNode:
    def __init__(self, dtype):
        self.dtype = dtype

        self.params = parameters.RosParams()
        self.logger = logger.RosLog()
        self.ackermann_msg_id = 0

    def start(self, name):
        rospy.init_node(name, anonymous=True)  # Initialize the node

        self.load_controller()
        self.setup_pub_sub()

        rate = rospy.Rate(25)
        self.inferred_pose = None
        print "Initialized"

        while not rospy.is_shutdown():
            if self.inferred_pose is not None:
                 next_ctrl = self.rhctrl.step(self.inferred_pose)
                 if next_ctrl is not None:
                     self.publish_ctrl(next_ctrl)
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
        rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.cb_goal, queue_size=1)
        rospy.Subscriber("/sim_car_pose/pose", PoseStamped, self.cb_pose, queue_size=10)

        self.rp_ctrls = rospy.Publisher(
            self.params.get_str(
                "~ctrl_topic",
                default="/vesc/high_level/ackermann_cmd_mux/input/nav_0"
            ),
            AckermannDriveStamped, queue_size=2
        )

    def cb_reset(self, msg):
        self.rhctrl.reset()

    def cb_odom(self, msg):
        self.inferred_pose = self.dtype(utils.rospose_to_posetup(msg.pose.pose))

    def cb_goal(self, msg):
        goal = self.dtype([
            msg.pose.position.x,
            msg.pose.position.y,
            utils.rosquaternion_to_angle(msg.pose.orientation)
        ])
        self.rhctrl.set_goal(goal)

    def cb_pose(self, msg):
        self.inferred_pose = self.dtype([
            msg.pose.position.x,
            msg.pose.position.y,
            utils.rosquaternion_to_angle(msg.pose.orientation)
        ])

    def publish_ctrl(self, ctrl):
        assert ctrl.size() == (2,)
        ctrlmsg = AckermannDriveStamped()
        ctrlmsg.header.stamp = rospy.Time.now()
        ctrlmsg.header.seq = self.ackermann_msg_id
        ctrlmsg.drive.speed = ctrl[0]
        ctrlmsg.drive.steering_angle = ctrl[1]
        self.rp_ctrls.publish(ctrlmsg)
        self.ackermann_msg_id += 1

    def get_model(self):
        mname = self.params.get_str("model_name", default="kinematic")
        if mname not in motion_models:
            self.logger.fatal("model '{}' is not valid".format(mname))
        return motion_models[mname](self.params, self.logger, self.dtype)

    def get_ctrl_gen(self):
        tgname = self.params.get_str("traj_gen_name", default="tl")
        if tgname not in trajgens:
            self.logger.fatal("ctrl_gen '{}' is not valid".format(tgname))
        return trajgens[tgname](self.params, self.logger, self.dtype)

    def get_map(self):
        srv_name = self.params.get_str("static_map", default="/static_map")
        self.logger.info("Waiting for map service")
        rospy.wait_for_service(srv_name)
        self.logger.info("Map service started")

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
        cfname = self.params.get_str("cost_fn_name", default="waypoints")
        if cfname not in cost_functions:
            self.logger.fatal("cost_fn '{}' is not valid".format(cfname))

        wrname = self.params.get_str("world_rep_name", default="simple")
        if wrname not in world_reps:
            self.logger.fatal("world_rep '{}' is not valid".format(wrname))

        map = self.get_map()
        wr = world_reps[wrname](self.params, self.logger, self.dtype, map)

        vfname = self.params.get_str("value_fn_name", default="simpleknn")
        if vfname not in value_functions:
            self.logger.fatal("value_fn '{}' is not valid".format(vfname))

        vf = value_functions[vfname](self.params, self.logger, self.dtype, map)

        return cost_functions[cfname](self.params, self.logger, self.dtype, wr, vf)
