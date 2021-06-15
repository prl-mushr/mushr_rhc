# Copyright (c) 2019, The Personal Robotics Lab, The MuSHR Team, The Contributors of MuSHR
# License: BSD 3-Clause. See LICENSE.md file in root directory.

import torch
import numpy as np
import yaml


class Waypoints:
    NPOS = 3  # x, y, theta

    def __init__(self, params, logger, dtype, map, world_rep):
        self.params = params
        self.logger = logger
        self.dtype = dtype
        self.map = map

        self.world_rep = world_rep

        self.viz_rollouts = self.params.get_bool("debug/flag/viz_rollouts", False)
        self.n_viz = self.params.get_int("debug/viz_rollouts/n", -1)
        self.print_stats = self.params.get_bool("debug/viz_rollouts/print_stats", False)

        self.reset()

    def reset(self):
        self.T = self.params.get_int("T", default=15)
        self.K = self.params.get_int("K", default=62)
        self.dist_w = self.params.get_float("cost_fn/dist_w", default=1.0)
        self.obs_dist_w = self.params.get_float("cost_fn/obs_dist_w", default=5.0)
        self.smoothing_discount_rate = self.params.get_float(
            "cost_fn/smoothing_discount_rate", default=0.04
        )
        self.bounds_cost = self.params.get_float("cost_fn/bounds_cost", default=100.0)

        self.obs_dist_cooloff = torch.arange(1, self.T + 1).mul_(2).type(self.dtype)

        self.discount = self.dtype(self.T - 1)

        self.discount[:] = 1 + self.smoothing_discount_rate
        self.discount.pow_(torch.arange(0, self.T - 1).type(self.dtype) * -1)
        self.world_rep.reset()

    def apply(self, poses, goal, path, car_pose):
        """
        Args:
        poses [(K, T, 3) tensor] -- Rollout of T positions
        goal  [(3,) tensor]: Goal position in "world" mode
        path nav_msgs.Path: Current path to the goal with orientation and positions
        car_pose geometry_msgs.PoseStamped: Current car position

        Returns:
        [(K,) tensor] costs for each K paths
        """
        assert poses.size() == (self.K, self.T, self.NPOS)
        assert goal.size() == (self.NPOS,)

        all_poses = poses.view(self.K * self.T, self.NPOS)

        # get all collisions (K, T, tensor)
        collisions = self.world_rep.check_collision_in_map(all_poses).view(
            self.K, self.T
        )
        collision_cost = collisions.sum(dim=1).mul(self.bounds_cost)

        # calculate lookahead
        distance_lookahead = 2.2

        # calculate closest index to car position
        diff = np.sqrt(
            ((path[:, 0] - car_pose[0]) ** 2) + ((path[:, 1] - car_pose[1]) ** 2)
        )
        index = np.argmin(diff)

        # iterate to closest lookahead to distance
        while index < len(path) - 1 and diff[index] < distance_lookahead:
            index += 1

        if abs(diff[index - 1] - distance_lookahead) < abs(
            diff[index] - distance_lookahead
        ):
            index -= 1

        x_ref, y_ref, theta_ref = path[index]

        cross_track_error = np.abs(
            -(poses[:, :, 0] - x_ref) * np.sin(theta_ref)
            + (poses[:, :, 1] - y_ref) * np.cos(theta_ref)
        )
        # take the sum of error along the trajs
        cross_track_error = torch.sum(cross_track_error, dim=1)

        along_track_error = np.abs(
            (poses[:, :, 0] - x_ref) * np.cos(theta_ref)
            + (poses[:, :, 1] - y_ref) * np.sin(theta_ref)
        )
        # take the sum of error along the trajs
        along_track_error = torch.sum(along_track_error, dim=1)

        # take the heading error from last pos in trajs
        heading_error = np.abs((poses[:, -1, 2] - theta_ref))

        # multiply weights
        cross_track_error *= self.params.get_int("cte_weight", default=1200)
        along_track_error *= self.params.get_int("ate_weight", default=1200)
        heading_error *= self.params.get_int("he_weight", default=100)

        result = cross_track_error.add(along_track_error).add(heading_error)

        colliding = collision_cost.nonzero()
        result[colliding] = 1000000000

        return result

    def set_goal(self, goal):
        self.goal = goal
        return True
