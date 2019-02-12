import os
import rospy
import threading
import cProfile

from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry, MapMetaData
from nav_msgs.srv import GetMap
from std_msgs.msg import Empty
from std_srvs.srv import Empty as SrvEmpty

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
    "dispersion": trajgen.Dispersion,
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
    def __init__(self, dtype, name):
        self.dtype = dtype

        self.params = parameters.RosParams()
        self.logger = logger.RosLog()
        self.ackermann_msg_id = 0

        self.reset_lock = threading.Lock()
        self.inferred_pose_lock = threading.Lock()
        self._inferred_pose = None

        self.goal_event = threading.Event()
        self.map_metadata_event = threading.Event()
        self.ready_event = threading.Event()
        self.events = [self.goal_event, self.map_metadata_event, self.ready_event]
        self.run = True

        rospy.init_node(name, anonymous=True, disable_signals=True)  # Initialize the node

        self.do_profile = self.params.get_bool("profile", default=False)

    def start_profile(self):
        if self.do_profile:
            self.logger.warn("Running with profiling")
            self.pr = cProfile.Profile()
            self.pr.enable()

    def end_profile(self):
        if self.do_profile:
            self.pr.disable()
            self.pr.dump_stats(os.path.expanduser('~/mushr_rhc_stats.prof'))

    def start(self):
        self.logger.info("Starting RHController")
        self.start_profile()
        self.setup_pub_sub()
        self.load_controller()

        self.ready_event.set()

        rate = rospy.Rate(50)
        self.logger.info("Initialized")

        # If we are trying to debug our rollouts, we only want to run
        # the loop on initial pose. This way of implementing it could be
        # changed, but for now this will get the job done
        if self.debug_rollouts:
            rospy.spin()

        while not rospy.is_shutdown() and self.run:
            next_ctrl = self.run_loop(self.inferred_pose())

            if next_ctrl is None:
                continue

            self.publish_ctrl(next_ctrl)
            # For experiments. If the car is at the goal, notify the
            # experiment tool
            if self.rhctrl.at_goal(self.inferred_pose()):
                self.expr_at_goal.publish(Empty())
                self.goal_event.clear()
            rate.sleep()

        self.end_profile()

    def run_loop(self, ip):
        self.goal_event.wait()
        if rospy.is_shutdown():
            return None
        with self.reset_lock:
            if ip is not None:
                return self.rhctrl.step(ip)

    def shutdown(self, signum, frame):
        rospy.signal_shutdown("SIGINT recieved")
        for ev in self.events:
            ev.set()
        self.run = False

    def load_controller(self):
        m = self.get_model()
        tg = self.get_trajgen(m)
        cf = self.get_cost_fn()

        self.rhctrl = librhc.MPC(self.params,
                                 self.logger,
                                 self.dtype,
                                 m, tg, cf)

    def setup_pub_sub(self):
        rospy.Service("~reset/soft", SrvEmpty, self.srv_reset_soft)
        rospy.Service("~reset/hard", SrvEmpty, self.srv_reset_hard)

        rospy.Subscriber("/move_base_simple/goal",
                         PoseStamped, self.cb_goal, queue_size=1)

        rospy.Subscriber("/map_metadata",
                         MapMetaData, self.cb_map_metadata, queue_size=1)

        if rospy.get_param("~use_sim_pose", default=False):
            rospy.Subscriber("/sim_car_pose/pose",
                             PoseStamped, self.cb_pose, queue_size=10)

        self.debug_rollouts = rospy.get_param("~debug_rollouts", default=False)
        if self.debug_rollouts:
            rospy.Subscriber("/initialpose",
                             PoseWithCovarianceStamped, self.cb_initialpose)

        if rospy.get_param("~use_odom_pose", default=True):
            rospy.Subscriber("/pf/pose/odom",
                             Odometry, self.cb_odom, queue_size=10)

        self.rp_ctrls = rospy.Publisher(
            self.params.get_str(
                "~ctrl_topic",
                default="/vesc/high_level/ackermann_cmd_mux/input/nav_0"
            ),
            AckermannDriveStamped, queue_size=2
        )

        # For the experiment framework, need indicators to listen on
        self.expr_at_goal = rospy.Publisher("/experiments/finished",
                                            Empty, queue_size=1)

    def srv_reset_hard(self, msg):
        '''
        Hard reset does a complete reload of the controller
        '''
        rospy.loginfo("Start hard reset")
        self.reset_lock.acquire()
        self.load_controller()
        self.reset_lock.release()
        rospy.loginfo("End hard reset")
        return []

    def srv_reset_soft(self, msg):
        '''
        Soft reset only resets soft state (like tensors). No dependencies or maps
        are reloaded
        '''
        rospy.loginfo("Start soft reset")
        self.reset_lock.acquire()
        self.rhctrl.reset()
        self.reset_lock.release()
        rospy.loginfo("End soft reset")
        return []

    def cb_odom(self, msg):
        self.set_inferred_pose(self.dtype(utils.rospose_to_posetup(msg.pose.pose)))

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

    def cb_initialpose(self, msg):
        ip = self.dtype(self.dtype(utils.rospose_to_posetup(msg.pose.pose)))

        self.run_loop(ip)

    def publish_ctrl(self, ctrl):
        assert ctrl.size() == (2,)
        ctrlmsg = AckermannDriveStamped()
        ctrlmsg.header.stamp = rospy.Time.now()
        ctrlmsg.header.seq = self.ackermann_msg_id
        ctrlmsg.drive.speed = ctrl[0]
        ctrlmsg.drive.steering_angle = ctrl[1]
        self.rp_ctrls.publish(ctrlmsg)
        self.ackermann_msg_id += 1

    def set_inferred_pose(self, ip):
        with self.inferred_pose_lock:
            self._inferred_pose = ip

    def inferred_pose(self):
        with self.inferred_pose_lock:
            return self._inferred_pose

    def get_model(self):
        mname = self.params.get_str("model_name", default="kinematic")
        if mname not in motion_models:
            self.logger.fatal("model '{}' is not valid".format(mname))

        return motion_models[mname](self.params, self.logger, self.dtype)

    def get_trajgen(self, model):
        tgname = self.params.get_str("trajgen_name", default="tl")
        if tgname not in trajgens:
            self.logger.fatal("trajgen '{}' is not valid".format(tgname))

        return trajgens[tgname](self.params, self.logger, self.dtype, model)

    def cb_map_metadata(self, msg):
        default_map_name = "default"
        map_file = self.params.get_str("map_file", default=default_map_name, global_=True)
        name = os.path.splitext(os.path.basename(map_file))[0]

        if name is default_map_name:
            rospy.logwarn("Default map name being used, will be corrupted on map change. " +
                          "To fix, set '/map_file' parameter with map_file location")

        x, y, angle = utils.rospose_to_posetup(msg.origin)
        self.map_data = types.MapData(
            name=name,
            resolution=msg.resolution,
            origin_x=x,
            origin_y=y,
            orientation_angle=angle,
            width=msg.width,
            height=msg.height,
            get_map_data=self.get_map
        )
        self.map_metadata_event.set()

    def get_map(self):
        srv_name = self.params.get_str("static_map", default="/static_map")
        self.logger.debug("Waiting for map service")
        rospy.wait_for_service(srv_name)
        self.logger.debug("Map service started")

        map_msg = rospy.ServiceProxy(srv_name, GetMap)().map
        return map_msg.data

    def get_cost_fn(self):
        cfname = self.params.get_str("cost_fn_name", default="waypoints")
        if cfname not in cost_functions:
            self.logger.fatal("cost_fn '{}' is not valid".format(cfname))

        wrname = self.params.get_str("world_rep_name", default="simple")
        if wrname not in world_reps:
            self.logger.fatal("world_rep '{}' is not valid".format(wrname))

        self.logger.debug("Waiting for map metadata")
        self.map_metadata_event.wait()
        self.logger.debug("Recieved map metadata")

        wr = world_reps[wrname](self.params, self.logger, self.dtype, self.map_data)

        vfname = self.params.get_str("value_fn_name", default="simpleknn")
        if vfname not in value_functions:
            self.logger.fatal("value_fn '{}' is not valid".format(vfname))

        vf = value_functions[vfname](self.params, self.logger, self.dtype, self.map_data)

        return cost_functions[cfname](self.params,
                                      self.logger,
                                      self.dtype,
                                      self.map_data, wr, vf)
