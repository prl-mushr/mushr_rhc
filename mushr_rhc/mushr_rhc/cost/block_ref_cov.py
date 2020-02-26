# Copyright (c) 2019, The Personal Robotics Lab, The MuSHR Team, The Contributors of MuSHR
# License: BSD 3-Clause. See LICENSE.md file in root directory.

import torch
import mushr_rhc.utils as utils
import threading

# send marker
import rospy
import tf.transformations
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Quaternion


def a2q(a):
    return Quaternion(*tf.transformations.quaternion_from_euler(0, 0, a))
# end send


class BlockRefTrajectoryCovariance:
    def __init__(
        self, params, logger, dtype, map, world_rep, value_fn, viz_rollouts_fn
    ):
        self.params = params
        self.logger = logger
        self.dtype = dtype
        self.map = map

        self.world_rep = world_rep
        self.value_fn = value_fn

        self.viz_rollouts_fn = viz_rollouts_fn

        self.DEBUG = rospy.Publisher("~current_marker", Marker, queue_size=1)
        self.BLOCKARRAY = rospy.Publisher("~block_viz", MarkerArray, queue_size=1)

        self.reset()

    def reset(self):
        self.T = self.params.get_int("T", default=15)
        self.K = self.params.get_int("K", default=62)
        self.NPOS = self.params.get_int("npos", default=3)

        self.dist_horizon = utils.get_distance_horizon(self.params)
        self.traj_lock = threading.RLock()
        with self.traj_lock:
            self.goal = None
        self.goal_threshold = self.params.get_float("xy_threshold", default=0.1)

        self.waypoint_lookahead = 0.2  # self.dist_horizon
        self.waypoint_idx_lookahead = 1  # 3

        self.dist_w = self.params.get_float("cost_fn/dist_w", default=1.0)

        self.world_rep.reset()

    def apply(self, poses, cov):
        """
        Args:
        poses [(K, T, 3) tensor] -- Rollout of T positions
        goal  [(3,) tensor]: Goal position in "world" mode
        cov   [(K, T, 3) tensor] -- Covariance of the pushes

        Returns:
        [(K,) tensor] costs for each K paths
        """
        assert poses.size() == (self.K, self.T, self.NPOS)

        with self.traj_lock:
            waypoint = self.get_waypoint(poses[0, 0, 3:5])

        dist_xy = (poses[:, 1:, 3:5] - waypoint[:2]).pow_(2)
        # currently, we ignore the first covariance, as we don't have a prior on the first state.
        # cov_log = cov.log()
        # cov_log[cov_log == float("-Inf")] = 0.0
        cov[:, 0, :] = 0.0001
        cov_sum_xy = torch.cumsum(cov[:, 1:, :], dim=1)
        # cov_sum_xy[cov_log_sum_xy == float("-Inf")] = 0.0

        # MAHALANOBIS over all points
        # mahalanobis = torch.sum(dist_xy[:, :, :2] / cov_sum_xy[:, :, :2], dim=2).sqrt()
        # traj_dists = torch.sum(mahalanobis, dim=1).div(self.T)

        mahalanobis = torch.sum(dist_xy[:, -1, :2] / cov_sum_xy[:, -1, :2], dim=1).sqrt()
        traj_dists = mahalanobis  # torch.sum(mahalanobis, dim=1).div(self.T)

        # send marker
        m = Marker()
        m.header.frame_id = "map"
        m.id = 0
        m.pose.position.x = waypoint[0]
        m.pose.position.y = waypoint[1]
        m.pose.orientation = a2q(waypoint[2])
        m.scale.x = 0.1
        m.scale.y = 0.1
        m.scale.z = 0.05
        m.color.r = 1.0
        m.color.a = 1.0
        self.DEBUG.publish(m)
        # end send

        result = traj_dists  # dist

        if self.viz_rollouts_fn:
            self.viz_rollouts_fn(
                result, poses, traj_dists=traj_dists
            )

            ma = MarkerArray()
            for i, p in enumerate(poses[:, self.T - 1, 3:6]):
                m = Marker()
                m.header.frame_id = "map"
                m.id = i
                m.type = 1
                m.pose.position.x = p[0]
                m.pose.position.y = p[1]
                m.pose.orientation = a2q(p[2])
                m.scale.x = 0.1
                m.scale.y = 0.1
                m.scale.z = 0.05
                m.color.r = float(i) / self.K
                m.color.a = 1.0
                ma.markers.append(m)
            self.BLOCKARRAY.publish(ma)

        return result

    def get_waypoint(self, state):
        dists = torch.norm(self.traj[:, :2] - state[:2], dim=1)
        idx = dists.argmin()
        idx += self.waypoint_idx_lookahead
        idx = min(idx, len(self.traj) - 1)
        return self.traj[idx]

    def set_trajectory(self, traj):
        """
        Args:
        goal [(3,) tensor] -- Goal in "world" coordinates
        """
        with self.traj_lock:
            self.traj = traj
            return True
            # return self.value_fn.set_goal(traj[-1])

    def dist_to_goal(self, state):
        with self.traj_lock:
            if self.traj is None:
                return False
            return self.traj[-1, :2].dist(state[3:5])

    def at_goal(self, state):
        """
        Args:
        state [(3,) tensor] -- Current position in "world" coordinates
        """
        with self.traj_lock:
            if self.traj is None:
                return False
            return self.dist_to_goal(state) < self.goal_threshold

    def get_desired_speed(self, desired_speed, state):
        return min(
            desired_speed,
            desired_speed * (self.dist_to_goal(state) / (self.dist_horizon)),
        )
