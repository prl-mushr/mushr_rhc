# Copyright (c) 2019, The Personal Robotics Lab, The MuSHR Team, The Contributors of MuSHR
# License: BSD 3-Clause. See LICENSE.md file in root directory.


class MPC:
    def __init__(self, params, logger, dtype, mvmt_model, trajgen, cost):
        self.dtype = dtype
        self.logger = logger
        self.params = params

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
        self.NPOS = self.params.get_int("npos", default=3)

        # Rollouts buffer, the main engine of our computation
        self.rollouts = self.dtype(self.K, self.T, self.NPOS)

        xy_thresh = self.params.get_float("xy_threshold", default=0.5)
        self.goal_threshold = self.dtype([xy_thresh, xy_thresh])
        self.desired_speed = self.params.get_float("trajgen/desired_speed", default=1.0)

        self.backwards = False

        if not init:
            self.trajgen.reset()
            self.kinematics.reset()
            self.cost.reset()

    def step(self, state):
        """
        Args:
        state [(3,) tensor] -- Current position in "world" coordinates
        """
        assert state.size() == (self.NPOS,)

        if self.cost.at_goal(state):
            return None, None

        v = self.cost.get_desired_speed(self.desired_speed, state)
        if self.backwards:
            v = -v

        trajs = self.trajgen.get_control_trajectories(v)
        assert trajs.size() == (self.K, self.T, 2)

        rollout_info = self.kinematics.rollout(state, trajs, self.rollouts)

        if rollout_info is not None:
            costs, self.backwards = self.cost.apply(self.rollouts, rollout_info)
        else:
            costs, self.backwards = self.cost.apply(self.rollouts)

        result, idx = self.trajgen.generate_control(trajs, costs)
        if idx is None:
            return result, None
        else:
            return result, self.rollouts[idx]

    def set_goal(self, goal):
        """
        Args:
        goal [(3,) tensor] -- Goal in "world" coordinates
        """
        assert goal.size() == (3,)
        return self.cost.set_goal(goal)

    def set_trajectory(self, traj):
        assert traj.size()[1] == 3
        return self.cost.set_trajectory(traj)
