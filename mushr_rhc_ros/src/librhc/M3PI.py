#!/usr/bin/env python

import time
import sys
import rospy
import rosbag
import numpy as np
import mcp_utils as Utils
from threading import Lock

import torch
import torch.utils.data
from torch.autograd import Variable
from torch.distributions import Categorical

from scipy import signal
from ReSample import ReSampler
import pickle

from nav_msgs.srv import GetMap
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker
from vesc_msgs.msg import VescStateStamped
from nav_msgs.msg import Path, Odometry
from m3pi.msg import Particle, ParticleArray, TLCosts
from std_msgs.msg import Float64, Empty
from geometry_msgs.msg import PoseStamped, PoseArray, PoseWithCovarianceStamped, PointStamped
from sensor_msgs.msg import Joy

import os.path


'''
These flags indicate the control strategy used by the MPPI controller.
'''
CM_MPPI = "CM_MPPI"
CM_M3PI = "CM_M3PI"
CM_TL = "CM_TL"


class MPPIController:

    def __init__(self):
        self.state_lock = Lock()
        self.SAMPLE_CONTROLS = 5
        self.INIT_MIN_VEL = -1.0  # TODO make sure these are right
        self.INIT_MAX_VEL = 1.0
        self.INIT_MIN_DEL = -0.34
        self.INIT_MAX_DEL = 0.34
        self.MIN_VEL = self.INIT_MIN_VEL  # TODO make sure these are right
        self.MAX_VEL = self.INIT_MAX_VEL
        self.MIN_DEL = self.INIT_MIN_DEL
        self.MAX_DEL = self.INIT_MAX_DEL
        self.BOUNDS_COST = 100.0
        self.MARKER_SIZE = 1.0
        self.SPEED_TO_ERPM_OFFSET = float(
            rospy.get_param("/vesc/speed_to_erpm_offset", 0.0))
        self.SPEED_TO_ERPM_GAIN = float(
            rospy.get_param(
                "/vesc/speed_to_erpm_gain",
                4614.0))
        self.STEERING_TO_SERVO_OFFSET = float(rospy.get_param(
            "/vesc/steering_angle_to_servo_offset", 0.5304))
        self.STEERING_TO_SERVO_GAIN = float(rospy.get_param(
            "/vesc/steering_angle_to_servo_gain", -1.2135))
        self.CAR_LENGTH = 0.37
        self.XY_THRESHOLD = float(rospy.get_param("~xy_threshold", 0.8))
        self.THETA_THRESHOLD = float(
            rospy.get_param(
                "~theta_threshold", np.pi))
        self.MODEL_NOISE = [0.005, 0.005, 0.001]

        self.config_controller()
        self.initialize_state()
        self.load_model_and_map()
        self.load_task()
        self.set_rollout_constants()
        self.initialize_buffers()
        self.initialize_ros_handles()

    def config_controller(self):
        # config
        self.control_mode = self.get_control_mode()
        self.viz_rollouts = bool(
            rospy.get_param(
                "~viz_rollouts",
                True))  # visualize rollouts
        self.viz_top_rollouts = bool(
            rospy.get_param(
                "~viz_top_rollouts",
                False))  # visualize everything else
        self.viz_waypoints = bool(rospy.get_param("~viz_waypoints", True))
        # visualize everything else
        self.viz = bool(rospy.get_param("~viz", True))
        self.cont_ctrl = bool(
            rospy.get_param(
                "~cont_ctrl",
                False))  # publish path continously
        self.covariance = bool(rospy.get_param("~covariance", False))
        self.sample_style = str(rospy.get_param("~sample_style", "low_var"))
        self.fixed_vel = bool(rospy.get_param("~fixed_vel", False))
        print("SETTING FIXED VEL {}".format(self.fixed_vel))
        self.debug = bool(rospy.get_param("~debug", False))
        self.step_control = bool(rospy.get_param("~step_control", False))

        # State related to varying traj opt strategy.
        self.tl_step_factor = int(rospy.get_param("~step_factor", 125))

    def initialize_state(self):
        self.time_index = 0

        # logging
        self.log_paths = False
        self.paths_list = []

        # Used for visualizing central path
        self.inferred_pose = None
        self.prev_inferred_pose = None
        self.pose_dot = None
        self.vizzing = False

        self.at_goal = False

        self.speed = 0
        self.steering_angle = 0
        self.prev_ctrl = None

        self._lambda = float(_lambda)

        self.lasttime = None

    def load_model_and_map(self, msg=None):
        # Loads nn model and map from server, takes optional Empty message
        # PyTorch / GPU data configuration
        # TODO
        # you should pre-allocate GPU memory when you can, and re-use it when
        # possible for arrays storing your controls or calculated MPPI costs,
        # etc
        model_name = rospy.get_param(
            "~nn_model", "/media/JetsonSSD/model3.torch")
        map_name = rospy.get_param(
            "/map/name", "identical_rooms").split('/')[-1].split('.')[0]

        self.model = torch.load(model_name)
        self.model.cuda()  # tell torch to run the network on the GPU
        self.model.eval()  # Model ideally runs faster in eval mode
        self.dtype = torch.cuda.FloatTensor
        self.softmax = torch.nn.Softmax()
        self.new_sigma = torch.Tensor([0.5, 0.1]).type(self.dtype)
        self.new_model_sigma = torch.Tensor(self.MODEL_NOISE).type(self.dtype)

        print("Loading:", model_name)
        print("Model:\n", self.model)
        print("Torch Datatype:", self.dtype)

        # Use the 'static_map' service (launched by MapServer.launch) to get
        # the map
        map_service_name = rospy.get_param("~static_map", "/static_map")
        print("Getting map from service: ", map_service_name)
        rospy.wait_for_service(map_service_name)
        # The map, will get passed to init of sensor model
        map_msg = rospy.ServiceProxy(map_service_name, GetMap)().map
        self.map_info = map_msg.info  # Save info about map for later use
        car_ratio = 3.5  # Ratio of car to extend in every direction TODO: project car into its actual orientation
        self.car_padding = long(
            (self.CAR_LENGTH / self.map_info.resolution) / car_ratio)
        self.map_angle = - \
            Utils.quaternion_to_angle(self.map_info.origin.orientation)
        self.map_c, self.map_s = np.cos(self.map_angle), np.sin(self.map_angle)
        print("Map Information:\n", self.map_info)

        # Create numpy array representing map for later use
        self.map_height = map_msg.info.height
        self.map_width = map_msg.info.width
        #PERMISSIBLE_REGION_FILE = '/home/nvidia/catkin_ws/src/m3pi/maps/permissible_region/' + map_name
        PERMISSIBLE_REGION_FILE = '/media/JetsonSSD/permissible_region/' + map_name
        print("Loading permissible region file (if exists): {}".format(
            PERMISSIBLE_REGION_FILE))
        if os.path.isfile(PERMISSIBLE_REGION_FILE + '.npy'):
            print("Already found permissible region file!")
            #self.permissible_region = np.load(PERMISSIBLE_REGION_FILE + '.npy')[::-1,:]
            self.permissible_region = np.load(
                PERMISSIBLE_REGION_FILE + '.npy')  # [::-1,:]
        else:
            print("MPPI: Computing permissible regions...")
            array_255 = np.array(
                map_msg.data).reshape(
                (map_msg.info.height, map_msg.info.width))
            self.permissible_region = np.zeros_like(array_255, dtype=bool)
            # Numpy array of dimension (map_msg.info.height,
            # map_msg.info.width),
            self.permissible_region[array_255 == 0] = 1
            # With values 0: not permissible, 1: permissible
            self.permissible_region = np.logical_not(
                self.permissible_region)  # 0 is permissible, 1 is not

            KERNEL_SIZE = 31  # 15 cm = 7 pixels = kernel size 15x15
            kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE))
            kernel /= kernel.sum()
            self.permissible_region = signal.convolve2d(
                self.permissible_region, kernel, mode='same') > 0  # boolean 2d array
            np.save(PERMISSIBLE_REGION_FILE, self.permissible_region)
        print("MPPI: finished loading perm. regions.")

        self.permissible_region = torch.from_numpy(
            self.permissible_region.astype(
                np.int)).type(
            torch.cuda.LongTensor)  # since we are using it as a tensor

    def load_task(self):
        waypoints_csv = rospy.get_param("~waypoints_csv", None)
        self.looping = rospy.get_param(
            "~looping", False)  # Allow for override possiblity
        self.waypoint_index = 0
        if waypoints_csv is not None:
            self.waypoints = torch.Tensor(
                Utils.load_csv_to_configs(waypoints_csv)).type(
                self.dtype)
        print("TASK LOADED: ", self.waypoints.cpu(), self.waypoints.shape)
        self.visited_waypoints = False

    def initialize_buffers(self):
        print("In initialize")
        # initialize these once
        self.goal_tensor = torch.zeros(3).type(self.dtype)
        self.ctrl = torch.zeros((self.T, 2)).cuda()

        self.sigma = self.new_sigma.repeat(self.T, self.K, 1)  # (T,K,2)
        self.model_sigma = self.new_model_sigma.repeat(
            self.T, self.K, 1)  # (T,K,2)
        self.noise = torch.Tensor(
            self.T, self.K, 2).type(
            self.dtype)  # (T,K,2)
        self.ctrl_noise_buf = torch.Tensor(
            self.T, self.K, 2).type(
            self.dtype)  # (T,K,2)
        if self.model_noise:
            self.model_noise_buf = torch.Tensor(
                self.T, self.K, 3).type(
                self.dtype)  # (T,K,3)

        # New cost fn
        self.all_poses = torch.Tensor(
            self.T * self.K,
            3).type(
            self.dtype)  # (T*K,3)
        self.t_vals = torch.Tensor(
            self.T, self.K).type(
            self.dtype)  # (T,K) (will view as T*K)
        for t in range(self.T):
            #self.t_vals[t,:] = 0.05 * (1.5 ** (0.3 * t))
            if self.control_mode == CM_MPPI or self.control_mode == CM_M3PI:
                if t < T - 1:
                    self.t_vals[t, :] = 0.01
                else:
                    self.t_vals[T - 1, :] = 0.8
            elif self.control_mode == CM_TL:
                self.t_vals[t, :] = 0.0 * (1.2 ** (0.5 * t))
            else:
                self.control_mode_not_recognized()

        self.t_vals = self.t_vals.view(self.T * self.K)  # (T*K)
        self.dist_to_goal = torch.Tensor(
            self.T * self.K,
            3).type(
            self.dtype)  # (T*K,3)
        self.euclidian_distance = torch.Tensor(
            self.T * self.K).type(self.dtype)  # (T*K,)
        self.temp_buf = torch.Tensor(
            self.K, self.T, 2).type(
            self.dtype)  # (K,T,2)
        self.cost_buf = torch.Tensor(
            self.T *
            self.K).type(
            self.dtype)  # (T*K,)

        self.poses = torch.Tensor(
            self.K, self.T, 3).type(
            self.dtype)  # (K,T,3)
        self.viz_poses = torch.Tensor(
            self.K, self.T, 3).type(
            self.dtype)  # (K,T,3)
        self.nn_input = torch.Tensor(self.K, 8).type(self.dtype)  # (K,8)
        self.pose_buf = torch.Tensor(self.K, 3).type(self.dtype)  # (K,3)
        self.central_path = torch.Tensor(self.T, 3).type(self.dtype)  # (T,3)
        self.running_cost = torch.zeros(self.K).type(self.dtype)  # (K,)

        self.curr_pose = torch.zeros(self.K, 3).type(self.dtype)  # (K,3)
        # Resampling P = num incoming particles, K = num particles/trajectories
        # to sample
        self.resample_pose_buf = torch.Tensor(
            self.P, 3).type(self.dtype)  # (P,3)
        self.resample_weight_buf = torch.Tensor(
            self.P).type(self.dtype)  # (P,)
        self.resample_pose_buf_map = torch.Tensor(
            self.P, 3).type(self.dtype)  # (P,3)
        self.good_ks = torch.Tensor(self.P).type(self.dtype)  # (P,)
        # self.last_pose = torch.zeros(self.K, 3).type(self.dtype).zero_()
        self.init_input = torch.zeros(self.K, 8).type(self.dtype)
        self.init_input_buf = torch.zeros(self.K, 8).type(self.dtype)

        self.recent_controls = np.zeros((self.SAMPLE_CONTROLS, 2))
        self.control_i = 0
        # control outputs
        self.msgid = 0

        if self.control_mode == CM_TL:
            self.tl_initialize()

        # visualization parameters
        self.num_viz_paths = 20
        if self.K < self.num_viz_paths:
            self.num_viz_paths = self.K

    def initialize_ros_handles(self):
        # We will publish control messages and a way to visualize a subset of our
        # rollouts, much like the particle filter
        self.ctrl_pub = rospy.Publisher(rospy.get_param("~ctrl_topic", "/vesc/high_level/ackermann_cmd_mux/input/nav_0"),
                                        AckermannDriveStamped, queue_size=2)
        self.path_pub = rospy.Publisher(
            "/mppi/paths", Path, queue_size=self.num_viz_paths)
        self.central_path_pub_m3pi = rospy.Publisher(
            "/m3pi/path_center", Path, queue_size=1)
        self.central_path_pub_mppi = rospy.Publisher(
            "/mppi/path_center", Path, queue_size=1)
        self.viz_waypoint_pub = rospy.Publisher(
            "/mppi/waypoints", Marker, queue_size=10)

        self.pose_pub = rospy.Publisher(
            "/mppi/pose", PoseStamped, queue_size=10)
        self.costs_pub = rospy.Publisher("/mppi/costs", TLCosts, queue_size=10)
        self.hparam_confirm_pub = rospy.Publisher(
            "/exp_tool/hparam_confirm", Empty, queue_size=1)

        print("Making callbacks")
        self.reset_sub = rospy.Subscriber("/mppi/reset",
                                          Empty, self.reset_cb, queue_size=1)
        self.reset_sub = rospy.Subscriber("/mppi/reset_model_and_map",
                                          Empty, self.load_model_and_map, queue_size=1)
        self.goal_sub = rospy.Subscriber("/pp/path_goal",
                                         PoseStamped, self.clicked_goal_cb, queue_size=1)
        # self.goal_sub = rospy.Subscriber("/pp/max_vel",
        #        Float64, self.max_vel_cb, queue_size=1)
        self.goal_sub_clicked = rospy.Subscriber("/move_base_simple/goal",
                                                 PoseStamped, self.clicked_goal_cb, queue_size=1)
        # self.pose_sub              = rospy.Subscriber("/pf/pose/odom",
        # PoseStamped, self.odom_cb, queue_size=10)
        if self.control_mode == CM_MPPI:
            self.pose_sub = rospy.Subscriber("/pf/pose/odom",
                                             Odometry, self.odom_cb, queue_size=10)
        elif self.control_mode == CM_M3PI:
            self.poses_sub = rospy.Subscriber("/pf/particles",
                                              ParticleArray, self.mppi_cb, queue_size=10)
            self.pose_sub = rospy.Subscriber("/pf/pose/odom",
                                             Odometry, self.odom_cb, queue_size=10)
        elif self.control_mode == CM_TL:
            self.odom_sub = rospy.Subscriber("/pf/pose/odom",
                                             Odometry, self.odom_cb, queue_size=10)
           # TODO Add particle subscription logic for control when time comes.
           # self.poses_sub  = rospy.Subscriber("/pf/viz/particles",
           #         PoseArray, self.mppi_cb, queue_size=10)

        self.joy_sub = rospy.Subscriber('/vesc/joy', Joy, self.joy_cb)

    def set_rollout_constants(self):
        print("Setting rollout constants!")
        self.model_noise = bool(rospy.get_param("~model_noise", True))
        self._lambda = float(rospy.get_param("~lambda", 0.62))
        self.new_sigma[0] = float(rospy.get_param("~sigma_v", 0.15))
        self.new_sigma[1] = float(rospy.get_param("~sigma_delta", 0.45))
        self.tl_step_factor = int(rospy.get_param("~tl_step_factor", 62))
        self.desired_speed = float(rospy.get_param("~desired_speed", 1.0))
        if self.fixed_vel:
            self.MAX_VEL = self.desired_speed
            self.MIN_VEL = self.desired_speed

        self.T = int(rospy.get_param("~T", 15))     # Length of rollout horizon
        # Number of particles sampled
        self.P = int(rospy.get_param("~P", 25))
        if (self.control_mode == CM_MPPI or self.control_mode == CM_M3PI):
            # Number of sampled rollouts.
            self.K = int(rospy.get_param("~mxpi_K", 4000))
        elif (self.control_mode == CM_TL):
            # Number of trajectories to sample in library.
            self.K = int(self.tl_step_factor)
        else:
            self.control_mode_not_recognized()
        print("T {} K {} P {}".format(self.T, self.K, self.P))

    def tl_initialize(self):
        self.cont_ctrl = False  # We want to execute control trajectory.

        step_size = (self.MAX_DEL - self.MIN_DEL) / (self.K - 1)
        deltas = torch.arange(
            self.MIN_DEL,
            self.MAX_DEL +
            step_size,
            step_size)

        # (1,) -> (T,K,1)
        self.ctrl_noise_buf[:, :, 0] = self.desired_speed
        self.ctrl_noise_buf[:, :, 1] = deltas.expand(
            (self.T, self.K))  # (K,) -> (T,K,1)

    def get_control_mode(self, current=False, default=CM_M3PI):
        if (current):
            return self.control_mode
        control_mode = str(rospy.get_param("~control_mode", default))
        return control_mode

    def control_mode_not_recognized(self):
        rospy.logerr(
            "Control mode {} not recognized. Shutting down....".format(
                self.control_mode))
        sys.exit(1)

    def reset_cb(self, msg):
        self.state_lock.acquire()
        print("MPPI: Resetting")
        self.config_controller()
        self.initialize_state()
        self.load_task()
        self.set_rollout_constants()
        self.initialize_buffers()
        print("MPPI: Control mode: {}".format(self.control_mode))
        print("MPPI: State reset to T, K, P, lambda",
              (self.T, self.K, self.P, self._lambda, self.new_sigma))
        print("MPPI: Reset complete")
        self.hparam_confirm_pub.publish(Empty())
        self.state_lock.release()

    def max_vel_cb(self, msg):
        speed = msg.data
        print "Max velocity set: {}".format(speed)
        self.MAX_VEL = speed
        self.MIN_VEL = -speed

    # TODO
    # You may want to debug your bounds checking code here, by clicking on a part
    # of the map and convincing yourself that you are correctly mapping the
    # click, and thus the goal pose, to accessible places in the map
    def clicked_goal_cb(self, msg):
        self.state_lock.acquire()
        config = [
            msg.pose.position.x,
            msg.pose.position.y,
            Utils.quaternion_to_angle(
                msg.pose.orientation)]
        print("config", config)
        goal_tensor_new = torch.Tensor(config).type(self.dtype)
        goal_tensor_long = goal_tensor_new.clone().unsqueeze(0)
        Utils.world_to_map_torch(
            goal_tensor_long,
            self.map_info,
            self.map_angle,
            self.map_c,
            self.map_s)
        goal_tensor_long = goal_tensor_long.long()[0]

        if goal_tensor_long[0] < 0 or goal_tensor_long[1] < 0 or goal_tensor_long[
                0] >= self.map_width or goal_tensor_long[1] >= self.map_height:
            print 'New goal outside of map bounds, not updating.'
            self.state_lock.release()
            return
         # if self.permissible_region[goal_tensor_long[1], goal_tensor_long[0]]:
         #   print 'New goal inside wall, not updating.'
         #   return

        if self.looping:
            self.waypoints[self.waypoint_index].copy_(goal_tensor_new)
            self.goal_tensor = goal_tensor_new
        else:
            self.goal_tensor = goal_tensor_new
            self.at_goal = False
        self.state_lock.release()
        print("SETTING Goal: ", self.goal_tensor.cpu().numpy())
        print("Waypoints ", self.waypoints.cpu().numpy())

    def cost(self, poses, goal, ctrl, noise):
        # print("cost Goal: " + str(goal))
        # input shapes: (K,T,3), (3,), (T,2), (T,K,2)

        self.running_cost.zero_()  # (K,)
        self.cost_buf.zero_()  # (T*K,)
        self.all_poses.zero_()  # (T*K,3)
        self.temp_buf.zero_()  # (K,T,2)
        self.dist_to_goal.zero_()  # (K,T,2)

        self.all_poses.add_(poses.view(self.K * self.T, 3))  # (T*K,3)

        # FIRST - get the pose cost
        self.dist_to_goal.copy_(self.all_poses).sub_(goal)  # (T*K,3)

        self.dist_to_goal[:, :2].pow_(2).sum(
            dim=1, out=self.euclidian_distance)  # eucl_dist = (T*K,)

        theta_distance = self.dist_to_goal[:, 2]  # (T*K,)
        Utils.clamp_angle_tensor_(theta_distance)
        theta_distance.abs_()
        self.cost_buf.add_((self.euclidian_distance.mul_(10)))  # (T*K)

        # SECOND - get the control cost, taking the abs
        # if we are using TL, lambda_ is 0, so this opereation is basically a no-op
        self.temp_buf.add_(
                ctrl.expand(self.K, self.T, 2)
            ).addcdiv_(
                 self._lambda, noise.transpose(0, 1), self.sigma.transpose(0, 1)
            ).abs_()  # (K,T,2)
        self.cost_buf.add_(
            self.temp_buf.sum(
                dim=2).mul_(1.0).view(
                self.T *
                self.K))  # (T*K)

        # THIRD - do a bounds check
        Utils.world_to_map_torch(
            self.all_poses,
            self.map_info,
            self.map_angle,
            self.map_c,
            self.map_s)  # cur pose is now in pixels (T*K,3)

        xs = self.all_poses[:, 0].long()
        ys = self.all_poses[:, 1].long()

        # (T*K,) with map value 0 or 1
        perm = (self.permissible_region[ys.long(), xs.long()])
        # (T*K,) with map value 0 or 1
        perm = (
            perm | self.permissible_region[ys.long() + self.car_padding, xs.long()])
        # (T*K,) with map value 0 or 1
        perm = (
            perm | self.permissible_region[ys.long() - self.car_padding, xs.long()])
        # (T*K,) with map value 0 or 1
        perm = (
            perm | self.permissible_region[ys.long(), xs.long() + self.car_padding])
        # (T*K,) with map value 0 or 1
        perm = (
            perm | self.permissible_region[ys.long(), xs.long() - self.car_padding])
        perm = perm.type(self.dtype)

        self.cost_buf.add_(
            value=self.BOUNDS_COST,
            other=perm)  # (T*K,) with map value 0 or 1

        self.running_cost.add_(
            self.cost_buf.view(
                self.K,
                self.T).sum(
                dim=1))  # (K,)

    def mppi(self, init_pose, init_input):
        self.state_lock.acquire()
        # init_pose (3,) [x, y, theta]
        # init_input (8,):
        #   0    1       2          3           4        5      6   7
        # xdot, ydot, thetadot, sin(theta), cos(theta), vel, delta, dt
        tx = time.time()
        t0 = time.time()
        #print("mppi: zero_: %4.5f ms" % ((time.time()-t0)*1000.0))
        #print("mppi: timing control: %4.5f ms" % ((time.time()-tx)*1000.0))
        dt = 0.1

        ta = time.time()
        self.running_cost.zero_()
        #print("mppi: zero_: %4.5f ms" % ((time.time()-ta)*1000.0))

        # convert pose into torch and one for each trajectory
        tb = time.time()
        self.pose_buf.copy_(init_pose)  # pose (K, 3)
        #print("mppi: pose_buf.copy_: %4.5f ms" % ((time.time()-tb)*1000.0))

        # create one input for each trajectory
        tc = time.time()
        Utils.clamp_angle_tensor_(init_input[:, 2])
        #print("mppi: clamp_angles of init_input: %4.5f ms" % ((time.time()-tc)*1000.0))
        tc = time.time()
        self.nn_input.copy_(init_input)  # nn_input (K, 8)
        #print("mppi: copy of init input into nn_input: %4.5f ms" % ((time.time()-tc)*1000.0))

        td = time.time()
        pose_dot = self.nn_input[:, :3]  # pose_dot (K, 3)
        #print("mppi: pose_dot: %4.5f ms" % ((time.time()-td)*1000.0))

        # MPPI should generate noise according to sigma
        te = time.time()

        # Create model noise
        if self.model_noise:
            torch.normal(0, self.model_sigma, out=self.model_noise_buf)

        if self.control_mode == CM_MPPI or self.control_mode == CM_M3PI:
            # Create noise for MPPI exploration
            torch.normal(0, self.sigma, out=self.noise)
        #print("mppi: normal: %4.5f ms" % ((time.time()-te)*1000.0))

        if self.viz and self.inferred_pose is not None:
            tf = time.time()
            # set trajectory 0 to have no noise when visualizing central path
            self.noise[:, 0, :] = 0
            #print("mppi: central no noise: %4.5f ms" % ((time.time()-tf)*1000.0))

        # combine noise with central control sequence
        tg = time.time()
        if self.control_mode == CM_MPPI or self.control_mode == CM_M3PI:
            self.ctrl_noise_buf.zero_()  # (T,K,2)
            if self.fixed_vel:
                self.noise[:, :, 0] = 0.1
                self.ctrl_noise_buf.add_(
                    self.ctrl.expand(
                        self.K,
                        self.T,
                        2).transpose(
                        0,
                        1)).add_(
                    self.noise)  # (T,K,2) + (T,K,2)
                # (1,) -> (T,K,1)
                self.ctrl_noise_buf[:, :, 0] = self.desired_speed
                self.ctrl[:, 0] = self.desired_speed
            else:
                self.ctrl_noise_buf.add_(
                    self.ctrl.expand(
                        self.K,
                        self.T,
                        2).transpose(
                        0,
                        1)).add_(
                    self.noise)  # (T,K,2) + (T,K,2)
                self.ctrl_noise_buf[:, :, 0].clamp_(self.MIN_VEL, self.MAX_VEL)
        self.ctrl_noise_buf[:, :, 1].clamp_(self.MIN_DEL, self.MAX_DEL)
        #print("mppi: transpose and more : %4.5f ms" % ((time.time()-tg)*1000.0))
        th = time.time()
        #print("mppi: clamp ctrl_noise_buf : %4.5f ms" % ((time.time()-th)*1000.0))

        # Perform rollouts with those controls from your current pose
        th = time.time()
        self.nn_input[:, 7] = dt  # dt
        #print("mppi: set dt : %4.5f ms" % ((time.time()-th)*1000.0))

        #print("mppi: remainder : %4.5f ms" % ((time.time()-th)*1000.0))
        #print("mppi: before t loop: %4.5f ms" % ((time.time()-t0)*1000.0))
        t1 = time.time()
        for t in xrange(self.T):

            # Update nn_inputs with new pose information
            self.nn_input[:, 0:3] = pose_dot  # xdot, ydot, thetadot
            torch.sin(self.pose_buf[:, 2],
                      out=self.nn_input[:, 3])  # sin(theta)
            torch.cos(self.pose_buf[:, 2],
                      out=self.nn_input[:, 4])  # cos(theta)

            self.nn_input[:, 5:7].copy_(self.ctrl_noise_buf[t])  # (K,2)

            # Call model to learn new pose_dot
            pose_dot = self.model(
                Variable(
                    self.nn_input,
                    requires_grad=False))  # (K, 3)
            pose_dot = pose_dot.data
            # Utils.clamp_angle_tensor_(pose_dot[:,2])
            self.pose_buf.add_(pose_dot)
            if self.model_noise:
                self.pose_buf.add_(
                    self.model_noise_buf[t, :, :])  # Update pose
            Utils.clamp_angle_tensor_(self.pose_buf[:, 2])

            # add new pose into poses
            # poses[:,t,:] (K,3) = pose (K,3)
            self.poses[:, t, :] = self.pose_buf.clone()
            # central_path[t,:] (1,3) = pose[0] (1,3)
            self.central_path[t, :] = self.pose_buf[0].clone()
        #print("mppi: t loop: %4.5f ms" % ((time.time()-t1)*1000.0))

        t2 = time.time()
        # (K,T,3), (3,), (T,2), (T,K,2) => (K,) (self.running_cost)
        self.cost(self.poses, self.goal_tensor, self.ctrl, self.noise)
        #print("mppi: cost: %4.5f ms" % ((time.time()-t2)*1000.0))
        if self.viz and self.control_mode == CM_TL:
            self.publish_tl_costs(init_pose)

        t3 = time.time()
        if (self.control_mode == CM_MPPI or self.control_mode == CM_M3PI):
                # Perform the MPPI weighting on your calculatd costs
                # Scale the added noise by the weighting and add to your
                # control sequence
            sorted, indices = torch.sort(self.running_cost, descending=True)
            idx = torch.Tensor(range(0,
                                     int(self.num_viz_paths**2),
                                     self.num_viz_paths)).type(torch.cuda.LongTensor)
            indices = indices[idx]
            costs, indices = torch.sort(self.running_cost, descending=False)
            #print("RUnning cost sorted {}".format(self.running_cost[indices]))
            beta, idx = self.running_cost.min(0)

            self.running_cost -= beta
            self.running_cost /= -self._lambda
            # print("RUnning cost - beta /= -lambda {}".format(self.running_cost[indices]))
            # Compute softmax of (-1/lambda)*(running_cost - min(running_cost))
            torch.exp(self.running_cost, out=self.running_cost)
            weights = self.running_cost
            w_sum = torch.sum(weights, -1)
            #print "w_sum shape: {}".format(w_sum.shape)
            if (w_sum != 0):
                # print("weights!")
                weights /= w_sum  # weights (K,)
            else:
                weights /= 0.00001

            # print("Weights {}".format(self.running_cost[indices]))
            weights = weights.expand(self.T, self.K)  # weights (T,K)
            weighted_vel_noise = weights * \
                self.noise[:, :, 0]  # (T,K) * (T,K) = (T,K)
            weighted_delta_noise = weights * \
                self.noise[:, :, 1]  # (T,K) * (T,K) = (T,K)

            # sum the weighted noise over all trajectories for each time step
            vel_noise_sum = torch.sum(weighted_vel_noise, dim=1)  # (T,)
            delta_noise_sum = torch.sum(weighted_delta_noise, dim=1)  # (T,)

            #print("self.sigma: ", self.sigma[:,0,:])
            # compute new covariance vector
            if (self.covariance):
                vel_diff = self.sigma[:, :, 0] - self.noise[:, :, 0]  # (T,K)
                delta_diff = self.sigma[:, :, 1] - self.noise[:, :, 1]  # (T,K)
                vel_cov = ((vel_diff * vel_diff.t()) * weights)  # (T,K)
                delta_cov = ((delta_diff * delta_diff.t()) * weights)  # (T,K)
                self.sigma[:,
                           :,
                           0] += torch.sum(vel_cov,
                                           dim=1).expand(self.K,
                                                         self.T).t()
                self.sigma[:,
                           :,
                           1] += torch.sum(delta_cov,
                                           dim=1).expand(self.K,
                                                         self.T).t()
                self.sigma[:, :, 0].clamp_(0.01, self.new_sigma[0])
                self.sigma[:, :, 1].clamp_(0.01, self.new_sigma[1])

            # update central control through time for vel and delta separately
            # self.ctrl # (T,2)
            if not self.fixed_vel:
                self.ctrl[:, 0] += vel_noise_sum  # (T,) += (T,)
                self.ctrl[:, 0].clamp_(self.MIN_VEL, self.MAX_VEL)

            # TODO: DEBUG NAN,
            # ALSO TODO: CHECK IN CODE PROGRESS
            self.ctrl[:, 1] += delta_noise_sum  # (T,) += (T,)
            self.ctrl[:, 1].clamp_(self.MIN_DEL, self.MAX_DEL)
            self.running_cost.zero_()
            weights.zero_()
        elif (self.control_mode == CM_TL):
            # Find index of best running cost trajectory
            weight, idx = self.running_cost.min(0)  # (K,) => (1,), (1,)
            # Select corresponding delta and velocity and update control
            self.ctrl.copy_(self.ctrl_noise_buf[:, idx, :])

        if self.viz and not self.vizzing:
            self.viz_poses.copy_(self.poses)

        #print("mppi: remainder: %4.5f ms" % ((time.time()-t2)*1000.0))

        # print("MPPI: %4.5f ms" % ((time.time()-t0)*1000.0))
        self.state_lock.release()
        return

    def get_control(self):
        # Apply the first control values, and shift your control trajectory
        run_ctrl = self.ctrl[0].clone().cpu()

        # shift all controls forward by 1, with last control replicated
        self.ctrl[:-1] = self.ctrl[1:]
        return run_ctrl

    def check_at_goal(self):
        # if self.at_goal or self.inferred_pose is None:
        #   return

        # TODO: if close to the goal, there's nothing to do
        # print(self.goal_tensor)
        difference_from_goal = (self.inferred_pose - self.goal_tensor).abs()
        xy_distance_to_goal = difference_from_goal[:2].norm()
        theta_distance_to_goal = difference_from_goal[2] % (2 * np.pi)
        if self.looping:
            dists = torch.norm(self.inferred_pose - self.waypoints, 2, dim=1)
            dists, indices = dists.sort()
            in_threshold = (
                xy_distance_to_goal < self.XY_THRESHOLD and theta_distance_to_goal < self.THETA_THRESHOLD)
            if in_threshold:  # and not self.visited_waypoint:
                print("HELLO")
                index = indices[1]
                self.visited_waypoint = True
                self.waypoint_index = int(index)
                self.goal_tensor.copy_(self.waypoints[self.waypoint_index])
           # elif not in_threshold:
           #   index = indices[0]
           #   self.visited_waypoint = False
           #   self.waypoint_index = int(index)
           #   self.goal_tensor.copy_(self.waypoints[self.waypoint_index])
        if (xy_distance_to_goal <
                self.XY_THRESHOLD and theta_distance_to_goal < self.THETA_THRESHOLD):
            print("WITHIN THRESHOLD")
            if not self.looping:
                self.at_goal = True
        return

    def odom_cb(self, msg):
        self.state_lock.acquire()
        self.inferred_pose = torch.Tensor(
            Utils.pose_to_config(
                msg.pose.pose)).type(
            self.dtype)
        self.state_lock.release()
        if self.prev_inferred_pose is None:
            self.prev_inferred_pose = self.inferred_pose.clone()
        self.check_at_goal()
        self.pose_dot = self.inferred_pose - self.prev_inferred_pose
        self.prev_inferred_pose = self.inferred_pose
        if self.control_mode == CM_MPPI or self.control_mode == CM_TL:
            self.mppi_cb(msg)

    def inferred_pose_cb(self, msg):
        self.state_lock.acquire()
        self.inferred_pose = torch.Tensor(
            Utils.posestamped_to_config(msg)).type(
            self.dtype)
        if self.prev_inferred_pose is None:
            self.prev_inferred_pose = self.inferred_pose.clone()
        self.pose_dot = self.inferred_pose - self.prev_inferred_pose
        self.prev_inferred_pose = self.inferred_pose
        self.state_lock.release()

    def particle_distance(self, point, vector):
        xs = vector[:, 0] - point[0]  # TODO: buf me up scotty
        ys = vector[:, 1] - point[1]
        theta = vector[:, 2] - point[2]
        return torch.sqrt(xs.pow_(2) + ys.pow_(2) + theta.pow_(2))

    def resample_poses(self):
        if self.inferred_pose is not None:
            t0 = time.time()
            self.resample_pose_buf_map.copy_(self.resample_pose_buf)
            Utils.world_to_map_torch(
                self.resample_pose_buf_map,
                self.map_info,
                self.map_angle,
                self.map_c,
                self.map_s)  # cur pose is now in pixels
            xs = self.resample_pose_buf_map[:, 0]
            ys = self.resample_pose_buf_map[:, 1]
            # (P,) with map value 0 or 1
            self.good_ks.copy_(self.permissible_region[ys.long(), xs.long()])
            if (self.good_ks.sum() == 0 or self.good_ks.sum() == self.P):
                inbounds_particles = self.resample_pose_buf[:]  # (P,3)
                inbounds_weights = self.resample_weight_buf[:]
            else:
                valid_indices = (self.good_ks == 0).nonzero().view(-1)
                # (P-n,3) where n is number removed particles
                inbounds_particles = self.resample_pose_buf[valid_indices]
                # (P-n,3)
                inbounds_weights = self.resample_weight_buf[valid_indices]
            # print("inbounds_particles")
            # print("istart", map(list, inbounds_particles.cpu()))
            inbounds_weights /= inbounds_weights.sum()
            if self.sample_style == "categorical":
                # print("categorical")
                q = 0.0  # Incorporate distance weights when re-sampling
                if q > 0:
                    # Inverse distance weight
                    dist_weights = 1.0 / \
                        self.particle_distance(
                            self.inferred_pose, inbounds_particles)
                    dist_weights /= dist_weights.sum()
                    sampler = Categorical(
                        inbounds_weights * (1 - q) + dist_weights * q)
                else:
                    sampler = Categorical(inbounds_weights)
                self.curr_pose.copy_(
                    inbounds_particles[sampler.sample_n(self.K)])  # (K,)
                #print("start", map(list, self.curr_pose.cpu()))
            else:
                # print("lowvar")
                sampler = ReSampler(inbounds_particles, inbounds_weights)
                self.curr_pose.copy_(
                    inbounds_particles[sampler.resample_low_variance(self.K)])
                #print("start", map(list, self.curr_pose.cpu()))
            #print("resample: %4.5f ms" % ((time.time()-t0)*1000.0))

    def mppi_cb(self, msg):
        for i in range(4):
            succ = self.mppi_cb_logic(msg)
            if not succ:
                break
        run_ctrl = None
        if self.cont_ctrl:
            run_ctrl = self.get_control().numpy()
            self.recent_controls[self.control_i] = run_ctrl
            self.control_i = (
                self.control_i + 1) % self.recent_controls.shape[0]
            pub_control = self.recent_controls.mean(0)
            self.speed = pub_control[0]
            self.steering_angle = pub_control[1]
        else:
            run_ctrl = self.get_control().numpy()
            self.speed = run_ctrl[0]
            self.steering_angle = run_ctrl[1]

    def mppi_cb_logic(self, msg):
        self.state_lock.acquire()
        t0 = time.time()
        if self.control_mode == CM_M3PI and len(msg.particles) != self.P:
            rospy.logerr(str(len(msg.poses)) +
                         " poses passed from particle filter. Expected " +
                         str(self.P))
            self.state_lock.release()
            return False

        if self.pose_dot is None:
            # if self.control_mode == CM_M3PI:
             #self.last_pose.copy_(torch.Tensor(map(Utils.pose_to_config, msg.poses)).type(self.dtype))
            self.lasttime = msg.header.stamp.to_sec()
            print("No pose_dot computed")
            self.state_lock.release()
            return False

        if self.control_mode == CM_M3PI:
            #Utils.poses_into_buffer(self.resample_pose_buf, msg.poses)
            particles, weights = Utils.parse_particle_array_msg(
                msg, self.dtype)
            self.resample_pose_buf.copy_(particles)  # (P,3) # TODO: uniform?
            self.resample_weight_buf.copy_(weights)  # (P,3) # TODO: uniform?
            self.resample_poses()
        elif self.control_mode == CM_MPPI:
            self.curr_pose.copy_(self.inferred_pose.expand(self.K, 3))
        elif self.control_mode == CM_TL:
            self.curr_pose.copy_(self.inferred_pose.expand(self.K, 3))
        else:
            self.control_mode_not_recognized()
        # self.curr_pose.copy_(torch.Tensor(map(Utils.pose_to_config,
        # msg.poses))) #(P,3) TODO: should we use non-blocking?
        if self.inferred_pose is not None:
            self.curr_pose[0] = self.inferred_pose
        theta = self.curr_pose[:, 2]

        # don't do any goal checking for testing purposes

        timenow = msg.header.stamp.to_sec()
        if timenow is not None and self.lasttime is not None:
            dt = timenow - self.lasttime
            if dt < 0.1:
                scale = 0.1 / dt
                self.pose_dot *= scale
                dt = 0.1
        else:
            dt = 0.1
        self.lasttime = timenow
        self.init_input_buf[:, 0:3] = self.pose_dot.expand(self.K, 3)  # (K,3)
        self.init_input_buf[:, 3] = torch.sin(theta)
        self.init_input_buf[:, 4] = torch.cos(theta)
        self.init_input_buf[:, 5] = 0
        self.init_input_buf[:, 6] = 0
        self.init_input_buf[:, 7] = dt

        self.init_input.copy_(self.init_input_buf)

        self.state_lock.release()

    def send_controls(self):
        self.state_lock.acquire()
        if self.cont_ctrl:
            speed = self.speed
            steer = self.steering_angle
            #self.time_index = (self.time_index + 1) % self.T
        else:
            speed = self.speed
            steer = self.steering_angle
        if speed != 0 and self.debug:
            print("Speed:", speed, "Steering:", steer)
        self.state_lock.release()
        ctrlmsg = AckermannDriveStamped()
        ctrlmsg.header.stamp = rospy.Time.now()
        ctrlmsg.header.seq = self.msgid
        ctrlmsg.drive.steering_angle = steer
        ctrlmsg.drive.speed = speed
        self.ctrl_pub.publish(ctrlmsg)
        self.msgid += 1

    # Publish some paths to RVIZ to visualize rollouts
    def visualize(self, poses, central_path):
        self.state_lock.acquire()
        self.vizzing = True
        # poses must be shape (self.num_viz_paths,T,3)
        if (self.looping and self.viz_waypoints):
            style = "waypoint"
            for i, waypoint in enumerate(self.waypoints):
                if i == self.waypoint_index:
                    style = "goal"
                marker = self.make_marker(waypoint, i, style)
                self.viz_waypoint_pub.publish(marker)
            self.viz_waypoints = False
        if (self.log_paths):
            indices = torch.arange(0, 4000, 100).type(torch.LongTensor)
            outgoing = self.poses.cpu()[indices, :, :].clone()
            outgoing[:, 0, :] = self.curr_pose.cpu()[indices]
            self.paths_list.append(outgoing)
        if (self.path_pub.get_num_connections() > 0
            and self.inferred_pose is not None
                and central_path is not None):
            print("vizzing sample of paths")
            frame_id = 'map'
            paths = []
            indices = range(0, self.num_viz_paths)
            if self.viz_top_rollouts:
                weights = self.running_cost
                print(weights)
                sorted, indices = torch.sort(weights, descending=True)
                idx = torch.Tensor(range(0,
                                         int(self.num_viz_paths**2),
                                         self.num_viz_paths)).type(torch.cuda.LongTensor)
                indices = indices[idx]
            for i in indices:
                pa = Path()
                pa.header = Utils.make_header(frame_id)
                # poses[i,:,:] has shape (T,3)
                pa.poses = map(Utils.particle_to_posestamped,
                               poses[i, :, :], [frame_id] * self.T)
                paths.append(pa)
            for pa in paths:
                self.path_pub.publish(pa)

        if (self.central_path_pub_mppi.get_num_connections() > 0
            and self.inferred_pose is not None
                and central_path is not None):

            if (self.control_mode == CM_MPPI or self.control_mode == CM_TL):
                # print("vizzing central path")
                frame_id = 'map'
                pa = Path()
                pa.header = Utils.make_header(frame_id)
                # central_path[:] has shape (T,3)
                pa.poses = map(
                    Utils.particle_to_posestamped,
                    central_path[:],
                    [frame_id] * self.T)
                self.central_path_pub_mppi.publish(pa)
            elif (self.control_mode == CM_M3PI):
                # print("vizzing central path")
                frame_id = 'map'
                pa = Path()
                pa.header = Utils.make_header(frame_id)
                # central_path[:] has shape (T,3)
                pa.poses = map(
                    Utils.particle_to_posestamped,
                    central_path[:],
                    [frame_id] * self.T)
                self.central_path_pub_m3pi.publish(pa)
            else:
                self.control_mode_not_recognized()

        self.vizzing = False
        self.state_lock.release()
        return True

    def make_marker(self, config, i, point_type):
        marker = Marker()
        marker.header = Utils.make_header('map')
        marker.ns = str(config)
        marker.id = i
        marker.type = Marker.CUBE
        marker.pose.position.x = config[0]
        marker.pose.position.y = config[1]
        marker.pose.orientation.w = 1
        marker.scale.x = self.MARKER_SIZE
        marker.scale.y = self.MARKER_SIZE
        marker.scale.z = self.MARKER_SIZE
        marker.color.a = 1.0
        if point_type == "waypoint":
            marker.color.b = 1.0
        else:
            marker.color.r = 1.0

        return marker

    def joy_cb(self, msg):
        # buttons=[A, B, X, Y, LB, RB, Back, Start, Logitech, Left joy, Right
        # joy]
        if msg.buttons[0]:
            print 'A button pressed'
            return
        if msg.buttons[1]:
            print 'B button pressed'
            return
        if msg.buttons[2]:
            print 'X button pressed'
            return
        if msg.buttons[3]:
            print 'Y button pressed'
            return

    def publish_tl_costs(self, init_pose):
        msg = Utils.create_tl_costs_msg(init_pose, self.running_cost)
        self.costs_pub.publish(msg)

    def shutdown_hook(self):
        if self.log_paths:
            name = "paths.pkl"
            with open(name, "wb") as f:
                pickle.dump(self.paths_list, f, pickle.HIGHEST_PROTOCOL)
            print(name + " saved!")


if __name__ == '__main__':

    T = int(rospy.get_param("/mppi/cov", False))
    T = int(rospy.get_param("/mppi/T", 30))
    K = int(rospy.get_param("/mppi/K", 2000))
    P = int(rospy.get_param("/mppi/P", 2000))
    sigma_v = float(rospy.get_param("/mppi/sigma_v", 0.15))
    sigma_theta = float(rospy.get_param("/mppi/sigma_theta", 0.45))
    _lambda = float(rospy.get_param("/mppi/lambda", .62))
    sigma = [sigma_v, sigma_theta]  # TODO: These values will need to be tuned

    rospy.init_node("mppi_control", anonymous=True)  # Initialize the node

    # run with ROS
    mp = MPPIController()
    rate = rospy.Rate(25)
    rospy.on_shutdown(mp.shutdown_hook)
    while not rospy.is_shutdown():
        # TODO: clean up this convoluted loop
        if mp.curr_pose is not None and mp.init_input is not None:
            if mp.step_control:
                raw_input("Press anything to step.")
            mp.mppi(mp.curr_pose, mp.init_input)
            if mp.viz and mp.viz_poses is not None:  # TODO: optimize visualization
                mp.visualize(mp.viz_poses, mp.central_path)

        mp.send_controls()
        rate.sleep()
