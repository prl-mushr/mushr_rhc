import numpy as np

class MPC:
    # Number of elements in the position vector
    NPOS = 3

    def __init__(self, params, logger, dtype, mvmt_model, traj_gen, cost):
        self.dtype = dtype
        self.logger = logger
        self.params = params
        self.goal = None

        self.traj_gen = traj_gen
        self.kinematics = mvmt_model
        self.cost = cost

        self.reset(init = True)

    def reset(self, init = False):
        self.T = self.params.get_int("T", default=15)
        self.K = self.params.get_int("K", default=62)

        # Rollouts buffer, the main engine of our computation
        self.rollouts = self.dtype(self.K, self.T, self.NPOS)

        xy_thresh = self.params.get_float("xy_threshold", default=0.8)
        th_thresh = self.params.get_float("theta_threshold", default=np.pi)
        self.goal_threshold = self.dtype([xy_thresh, xy_thresh, th_thresh])

        if not init:
            self.traj_gen.reset()
            self.kinematics.reset()
            self.cost.reset()

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
        result = self.traj_gen.generate_control(trajs, costs)[0]
        import rospy
        rospy.loginfo_throttle(1, "Controll applied: " + str(result))
        return result

    def set_goal(self, goal):
        self.goal = goal
        self.cost.value_fn.set_goal(goal)

    def _at_goal(self, state):
        assert self.goal is not None
        dist = self.goal.sub(state).abs_()
        return dist.lt(self.goal_threshold).min() == 1
