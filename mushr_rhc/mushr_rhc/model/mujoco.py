import rospy

import mushr_rhc
import tf.transformations

from ackermann_msgs.msg import AckermannDrive
from mushr_mujoco_ros.msg import AckermannDriveArray

from mushr_mujoco_ros.srv import Rollout


def rosquaternion_to_angle(q):
    """Convert a quaternion _message_ into an angle in radians.
    The angle represents the yaw.
    This is not just the z component of the quaternion."""
    x, y, z, w = q.x, q.y, q.z, q.w
    _, _, yaw = tf.transformations.euler_from_quaternion((x, y, z, w))
    return yaw


def rospose_to_posetup(posemsg):
    x = posemsg.position.x
    y = posemsg.position.y
    th = rosquaternion_to_angle(posemsg.orientation)
    return x, y, th


class MujocoSim:
    NCTRL = 2

    def __init__(self, params, logger, dtype):
        self.logger = logger
        self.params = params
        self.dtype = dtype

        self.reset()

    def set_k(self, k):
        self.logger.fatal("Can't change the size of K for the mujoco sim model")

    def connect_rollout_service(self):
        self.logger.info("Waiting for rollout service")
        self.logger.info(rospy.resolve_name("mujoco_rollout"))
        rospy.wait_for_service("mujoco_rollout")
        self.logger.info("Found rollout service")
        self.mj_rollout = rospy.ServiceProxy("mujoco_rollout", Rollout, persistent=True)

    def reset(self):
        self.K = self.params.get_int("K", default=62)
        self.T = self.params.get_int("T", default=21)
        self.NPOS = self.params.get_int("npos", default=3)

        time_horizon = mushr_rhc.utils.get_time_horizon(self.params)
        self.dt = time_horizon / self.T
        self.connect_rollout_service()

    def rollout(self, state, trajs, rollouts):
        ctrls = []
        for t in trajs:
            ada = AckermannDriveArray()
            for c in t:
                ada.data.append(AckermannDrive(speed=c[0], steering_angle=c[1]))
            ctrls.append(ada)
        try:
            # call svc
            res = self.mj_rollout(self.K, self.T, self.dt, ctrls)
        except Exception as e:
            self.logger.info("Error getting rollouts: " + str(e))
            self.connect_rollout_service()

        for k in range(self.K):
            car_poses = res.car_poses[k]
            block_poses = res.block_poses[k]
            for t in range(self.T):
                x, y, theta = rospose_to_posetup(car_poses.poses[t])
                rollouts[k, t, 0] = x
                rollouts[k, t, 1] = y
                rollouts[k, t, 2] = theta

                x, y, theta = rospose_to_posetup(block_poses.poses[t])
                rollouts[k, t, 3] = x
                rollouts[k, t, 4] = y
                rollouts[k, t, 5] = theta
