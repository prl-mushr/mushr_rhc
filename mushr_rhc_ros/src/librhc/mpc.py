import numpy as np

class MPC:
    # Number of elements in the position vector
    NPOS = 3

    def __init__(self, params, logger, dtype, mvmt_model, traj_gen, cost):
        self.T = params.get_int("T", default=15)
        self.K = params.get_int("K", default=62)
        self.dtype = dtype
        self.goal = None

        # Rollouts buffer, the main engine of our computation
        self.rollouts = self.dtype(self.K, self.T, self.NPOS)

        xy_thresh = params.get_float("xy_threshold", default=0.8)
        th_thresh = params.get_float("theta_threshold", default=np.pi)
        self.goal_threshold = self.dtype([xy_thresh, xy_thresh, th_thresh])

        self.traj_gen = traj_gen
        self.kinematics = mvmt_model
        self.cost = cost
        self.logger = logger

    # TODO: Return None when we are at the goal
    def step(self, state):
        """
        Args:
          state (3, tensor): current position
        """
        assert state.size() == (3,)
        if self.goal is None:
            return None

        if self._at_goal(state):
            return None

        self.rollouts.zero_()

        # For each K trial, the first position is at the current position
        self.rollouts[:, 0] = state.expand_as(self.rollouts[:, 0])

        trajs = self.traj_gen.get_control_trajectories()
        assert trajs.size() == (self.K, self.T, 2)

        for t in range(1, self.T):
            cur_x = self.rollouts[:, t - 1]
            self.rollouts[:, t] = self.kinematics.apply(cur_x, trajs[:, t - 1])

        costs = self.cost.apply(self.rollouts, self.goal)

        return self.traj_gen.generate_control(trajs, costs)[0]

    def set_goal(self, goal):
        self.logger.warn("Setting goal" % goal)
        self.goal = goal

    def _at_goal(self, state):
        assert self.goal is not None
        dist = self.goal.sub(state).abs_()
        return dist.lt(self.goal_threshold).min() == 1
