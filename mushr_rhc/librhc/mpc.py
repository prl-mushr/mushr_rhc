# Copyright (c) 2019, The Personal Robotics Lab, The MuSHR Team, The Contributors of MuSHR
# License: BSD 3-Clause. See LICENSE.md file in root directory.

import threading

import librhc.utils as utils


class MPC:
    # Number of elements in the position vector
    NPOS = 3

    def __init__(self, params, logger, dtype, mvmt_model, trajgen, cost):
        self.dtype = dtype
        self.logger = logger
        self.params = params
        self.goal = None

        self.trajgen = trajgen
        self.kinematics = mvmt_model
        self.cost = cost

        self.reset(init=True)

    def reset(self, init=False):
        """
        Args:
        init [bool] -- whether this is being called by the init function
        """
        self.T = self.params.get_int("T", default=15)
        self.K = self.params.get_int("K", default=62)

        # Rollouts buffer, the main engine of our computation
        self.rollouts = self.dtype(self.K, self.T, self.NPOS)

        xy_thresh = self.params.get_float("xy_threshold", default=0.5)
        self.goal_threshold = self.dtype([xy_thresh, xy_thresh])
        self.desired_speed = self.params.get_float("trajgen/desired_speed", default=1.0)
        self.dist_horizon = utils.get_distance_horizon(self.params)

        self.goal_lock = threading.Lock()
        with self.goal_lock:
            self.goal = None

        if not init:
            self.trajgen.reset()
            self.kinematics.reset()
            self.cost.reset()

    def step(self, state):
        """
        Args:
        state [(3,) tensor] -- Current position in "world" coordinates
        """
        assert state.size() == (3,)

        if self.at_goal(state):
            return None, None

        with self.goal_lock:
            g = self.goal

        self.rollouts.zero_()

        # For each K trial, the first position is at the current position
        self.rollouts[:, 0] = state.expand_as(self.rollouts[:, 0])

        v = min(
            self.desired_speed,
            self.desired_speed * (self.dist_to_goal(state) / (self.dist_horizon)),
        )
        trajs = self.trajgen.get_control_trajectories(v)
        assert trajs.size() == (self.K, self.T, 2)

        for t in range(1, self.T):
            cur_x = self.rollouts[:, t - 1]
            self.rollouts[:, t] = self.kinematics.apply(cur_x, trajs[:, t - 1])

        costs = self.cost.apply(self.rollouts, g)
        result, idx = self.trajgen.generate_control(trajs, costs)
        return result, self.rollouts[idx]

    def set_goal(self, goal):
        """
        Args:
        goal [(3,) tensor] -- Goal in "world" coordinates
        """
        assert goal.size() == (3,)

        with self.goal_lock:
            self.goal = goal
            return self.cost.value_fn.set_goal(goal)

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
        dist = self.goal[:2].sub(state[:2]).abs_()
        return dist.lt(self.goal_threshold).min() == 1
