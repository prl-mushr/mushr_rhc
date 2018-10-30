import rospy

from nav_msgs.msg import Odometry
from nav_msgs.srv import GetMap
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseStamped

import logger
import parameters
import utils

import librhc
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
        self.logger = logger.RosLogger()

    def start(self, name):
        rospy.init_node(name, anonymous=True)  # Initialize the node

        self.load_controller()
        self.setup_pub_sub()

        rate = rospy.Rate(25)

        while not rospy.is_shutdown():
            next_ctrl = self.rhctrl.step(self.inferred_pose)
            self.publish_ctrl(next_ctrl)
            rate.sleep()

    def load_controller(self):
        m = self.get_model(params)
        cg = self.get_ctrl_generator(params)
        cf = self.get_cost_function(params)

        self.rhctrl = libmushr_rhc.MPC(self.params, self.logger,
                            self.dtype, m, cg, cf)

    def setup_pub_sub(self):
        rospy.Subscriber("/rhc/reset", Empty, self.cb_reset, queue_size=1)
        rospy.Subscriber("/pf/pose/odom", Odometry, self.cb_odom, queue_size=10)
        rospy.Subscriber("/pp/path_goal", PoseStamped, self.cb_goal, queue_size=1)

        self.rp_ctrls = rospy.Publisher(
            rospy.get_param(
                "~ctrl_topic",
                "/vesc/high_level/ackermann_cmd_mux/input/nav_0"
            ),
            AckermannDriveStamped, queue_size=2
        )

    def cb_reset(self, msg):
        self.load_controller()

    def cb_odomb(self, msg):
        self.inferred_pose = self.dtype(utils.pose_to_config(msg.pose.pose))

    def cb_goal(self, msg):
        goal = self.dtype([
            msg.pose.position.x,
            msg.pose.position.y,
            utils.rosquaternion_to_angle(msg.pose.orientation)
        ])
        self.rhctrl.model.set_goal(goal)

    def publish_ctrl(self, ctrl):
        ctrlmsg = AckermannDriveStamped()
        ctrlmsg.header.stamp = rospy.Time.now()
        ctrlmsg.header.seq = self.ackermann_msg_id
        ctrlmsg.drive.steering_angle = ctrl[0]
        ctrlmsg.drive.speed = ctrl[1]
        self.rp_ctrls.publish(ctrlmsg)
        self.ackermann_msg_id += 1

    def get_model(self):
        mname = self.params.get_str("model", "kinematic")
        if mname not in models:
            self.logger.fatal("model '{}' is not valid".format(mname))
        return models[mname](params, logger, FLOAT_TENSOR)

    def get_ctrl_gen(self):
        tgname = self.params.get_str("ctrl_generator", "tl")
        if cgname not in trajgens:
            self.logger.fatal("ctrl_gen '{}' is not valid".format(tgname))
        return trajgens[tgname](params, logger, FLOAT_TENSOR)

    def get_map(self):
        srv_name = str(rospy.get_param("static_map"))
        rospy.wait_for_service(srv_name)

        map_msg = rospy.ServiceProxy(srv_name, GetMap)().map

        return types.MapData(
            resolution = map_msg.info.resolution,
            origin_x = map_msg.info.origin.x,
            origin_y = map_msg.info.origin.y,
            orientation_angle = map_msg.info.origin.angle,
            width = map_msg.info.width,
            height = map_msg.info.height,
            data = map_msg.data
        )

    def get_cost_fn(self):
        cfname = self.params.get_str("cost_fn", "rename")
        if cfname not in cost_functions:
            self.logger.fatal("cost_fn '{}' is not valid".format(cfname))

        wrname = self.params.get_str("world_rep", "simple")
        if wrname not in world_reps:
            self.logger.fatal("world_rep '{}' is not valid".format(wrname))

        wr = world_reps[wrname](params, logger, FLOAT_TENSOR, get_map())

        return cost_functions[cfname](params, logger, FLOAT_TENSOR, wr)
