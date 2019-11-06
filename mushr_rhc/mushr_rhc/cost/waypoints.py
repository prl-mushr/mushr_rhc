# Copyright (c) 2019, The Personal Robotics Lab, The MuSHR Team, The Contributors of MuSHR
# License: BSD 3-Clause. See LICENSE.md file in root directory.

import torch
import mushr_rhc.utils as utils
import threading


class Waypoints:
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
        self.dist_horizon = utils.get_distance_horizon(self.params)

        self.reset()

    def reset(self):
        self.T = self.params.get_int("T", default=15)
        self.K = self.params.get_int("K", default=62)
        self.NPOS = self.params.get_int("npos", default=3)

        self.goal_lock = threading.RLock()
        with self.goal_lock:
            self.goal = None
        self.goal_threshold = self.params.get_float("xy_threshold", default=0.5)

        self.dist_w = self.params.get_float("cost_fn/dist_w", default=1.0)
        self.obs_dist_w = self.params.get_float("cost_fn/obs_dist_w", default=5.0)
        self.cost2go_w = self.params.get_float("cost_fn/cost2go_w", default=1.0)
        self.smoothing_discount_rate = self.params.get_float(
            "cost_fn/smoothing_discount_rate", default=0.04
        )
        self.bounds_cost = self.params.get_float("cost_fn/bounds_cost", default=100.0)

        self.obs_dist_cooloff = torch.arange(1, self.T + 1).mul_(2).type(self.dtype)

        self.discount = self.dtype(self.T - 1)

        self.discount[:] = 1 + self.smoothing_discount_rate
        self.discount.pow_(torch.arange(0, self.T - 1).type(self.dtype) * -1)
        self.world_rep.reset()

    def apply(self, poses):
        """
        Args:
        poses [(K, T, 3) tensor] -- Rollout of T positions
        goal  [(3,) tensor]: Goal position in "world" mode

        Returns:
        [(K,) tensor] costs for each K paths
        """
        assert poses.size() == (self.K, self.T, self.NPOS)

        with self.goal_lock:
            goal = self.goal

        assert goal.size() == (self.NPOS,)

        all_poses = poses.view(self.K * self.T, self.NPOS)

        # use terminal distance (K, tensor)
        cost2go = self.value_fn.get_value(poses[:, self.T - 1, :]).mul(self.cost2go_w)

        # get all collisions (K, T, tensor)
        collisions = self.world_rep.check_collision_in_map(all_poses).view(
            self.K, self.T
        )
        collision_cost = collisions.sum(dim=1).mul(self.bounds_cost)

        obstacle_distances = self.world_rep.distances(all_poses).view(self.K, self.T)
        obstacle_distances[:].mul_(self.obs_dist_cooloff)

        obs_dist_cost = obstacle_distances[:].sum(dim=1).mul(self.obs_dist_w)

        # reward smoothness by taking the integral over the rate of change in poses,
        # with time-based discounting factor
        smoothness = (
            ((poses[:, 1:, 2] - poses[:, : self.T - 1, 2]))
            .abs()
            .mul(self.discount)
            .sum(dim=1)
        )

        result = cost2go.add(collision_cost).add(obs_dist_cost).add(smoothness)

        # filter out all colliding trajectories
        colliding = collision_cost.nonzero()
        result[colliding] = 1000000000

        if self.viz_rollouts_fn:
            self.viz_rollouts_fn(
                result, cost2go, collision_cost, obs_dist_cost, smoothness
            )

        return result

    def set_goal(self, goal):
        """
        Args:
        goal [(3,) tensor] -- Goal in "world" coordinates
        """
        assert goal.size() == (3,)

        with self.goal_lock:
            self.goal = goal
            return self.value_fn.set_goal(goal)

    def dist_to_goal(self, state):
        with self.goal_lock:
            if self.goal is None:
                return False
            return self.goal[:2].dist(state[:2])

    def at_goal(self, state):
        """
        Args:
        state [(3,) tensor] -- Current position in "world" coordinates
        """
        with self.goal_lock:
            if self.goal is None:
                return False
            return self.dist_to_goal(state) < self.goal_threshold

    def get_desired_speed(self, desired_speed, state):
        return min(
            desired_speed,
            desired_speed * (self.dist_to_goal(state) / (self.dist_horizon)),
        )
