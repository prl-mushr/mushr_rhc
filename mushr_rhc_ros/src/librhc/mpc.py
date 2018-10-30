import np

class MPC:
    # Number of elements in the position vector
    NPOS = 3

    def __init__(self, params, logger, dtype, mvmt_model, traj_gen, cost):
        self.T = params.get_int("T")
        self.K = params.get_int("K")
        self.dtype = dtype

        # Rollouts buffer, the main engine of our computation
        self.rollouts = self.dtype(K, T, NPOS)

        xy_thresh = params.get_float("~xy_threshold", default=0.8)
        th_thresh = params.get_float("~xy_threshold", default=np.pi)
        self.goal_threshold = self.dtype([xy_thresh, xy_thresh, th_thresh])

        self.traj_gen = traj_gen
        self.model = mvmt_model
        self.cost = cost
        self.logger = logger

    # TODO: Return None when we are at the goal
    def step(self, state):
        """
        Args:
          state (1,3 tensor): current position
        """
        if self._at_goal(state):
            return None

        self.rollouts.zero_()

        # For each K trial, the first position is at the current position
        self.rollouts[:, 0] = state.expand_as(self.rollouts[:, 0])

        trajs = traj_gen.get_control_trajectories()

        for t in range(1, self.T):
            cur_x = self.rollouts[:, t - 1]
            self.rollouts[:, t] = self.kinematics.apply(cur_x, trajs[:, t - 1])

        costs = self.cost.apply(self.rollouts, self.goal)

        return traj_gen.generate_control(ctrls, costs)

    def set_goal(self, goal):
        self.goal = goal

    def _at_goal(self, state):
        dist = self.goal.sub(state).abs_()
        return dist.lt(self.goal_thresholds).min() == 1
