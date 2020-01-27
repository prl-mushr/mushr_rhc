# Copyright (c) 2019, The Personal Robotics Lab, The MuSHR Team, The Contributors of MuSHR
# License: BSD 3-Clause. See LICENSE.md file in root directory.

import mushr_rhc


class Kinematics:
    EPSILON = 1e-5
    NCTRL = 2

    def __init__(self, params, logger, dtype):
        self.logger = logger
        self.params = params
        self.dtype = dtype

        self.reset()

    def reset(self):
        self.set_k(self.params.get_int("K", default=62))
        self.T = self.params.get_int("T", default=21)
        self.NPOS = self.params.get_int("npos", default=3)

    def set_k(self, k):
        """
        In some instances the internal buffer size needs to be changed. This easily facilitates this change

        Args:
        k [int] -- Number of rollouts
        """
        self.K = k
        self.wheel_base = self.params.get_float("model/wheel_base", default=0.3)

        time_horizon = mushr_rhc.utils.get_time_horizon(self.params)
        T = self.params.get_int("T", default=15)
        self.dt = time_horizon / T

        # OLD
        # self.sin2beta = self.dtype(self.K)
        self.beta = self.dtype(self.K)
        self.deltaTheta = self.dtype(self.K)
        self.deltaX = self.dtype(self.K)
        self.deltaY = self.dtype(self.K)
        self.sin_t = self.dtype(self.K)
        self.cos_t = self.dtype(self.K)
        self.sin_t_1 = self.dtype(self.K)
        self.cos_t_1 = self.dtype(self.K)

    def rollout(self, state, trajs, rollouts):
        rollouts.zero_()

        # For each K trial, the first position is at the current position
        rollouts[:, 0] = state.expand_as(rollouts[:, 0])

        for t in range(1, self.T):
            cur_x = rollouts[:, t - 1]
            rollouts[:, t] = self.apply(cur_x, trajs[:, t - 1])

    def apply(self, pose, ctrl):
        """
        Args:
        pose [(K, NPOS) tensor] -- The current position in "world" coordinates
        ctrl [(K, NCTRL) tensor] -- Control to apply to the current position
        Return:
        [(K, NCTRL) tensor] The next position given the current control
        """
        assert pose.size() == (self.K, self.NPOS)
        assert ctrl.size() == (self.K, self.NCTRL)

        self.beta.copy_(ctrl[:, 1]).tan_().div_(2.0).atan_().add_(self.EPSILON)
        sinbeta = self.beta.sin()

        self.deltaTheta.copy_(sinbeta).mul_(2).mul_(ctrl[:, 0] * .9).div_(
            self.wheel_base).mul_(self.dt)

        nextpos = self.dtype(self.K, 3)
        nextpos.copy_(pose)
        nextpos[:, 2].add_(self.deltaTheta)

        self.sin_t.copy_(pose[:, 2]).add_(self.beta).sin_()
        self.cos_t.copy_(pose[:, 2]).add_(self.beta).cos_()

        self.sin_t_1.copy_(nextpos[:, 2]).add_(self.beta).sin_()
        self.cos_t_1.copy_(nextpos[:, 2]).add_(self.beta).cos_()

        self.deltaX.copy_(self.sin_t_1).sub_(self.sin_t).mul_(
            self.wheel_base).div_(2).div_(sinbeta)

        self.deltaY.copy_(self.cos_t).sub_(self.cos_t_1).mul_(
            self.wheel_base).div_(2).div_(sinbeta)

        nextpos[:, 0].add_(self.deltaX)
        nextpos[:, 1].add_(self.deltaY)

        return nextpos

    def OLDapply(self, pose, ctrl):
        """
        Args:
        pose [(K, NPOS) tensor] -- The current position in "world" coordinates
        ctrl [(K, NCTRL) tensor] -- Control to apply to the current position
        Return:
        [(K, NCTRL) tensor] The next position given the current control
        """
        assert pose.size() == (self.K, self.NPOS)
        assert ctrl.size() == (self.K, self.NCTRL)

        self.sin2beta.copy_(ctrl[:, 1]).tan_().mul_(0.5).atan_().mul_(2.0).sin_().add_(
            self.EPSILON
        )

        self.deltaTheta.copy_(ctrl[:, 0]).div_(self.wheel_base).mul_(
            self.sin2beta
        ).mul_(self.dt)

        self.sin.copy_(pose[:, 2]).sin_()
        self.cos.copy_(pose[:, 2]).cos_()

        self.deltaX.copy_(pose[:, 2]).add_(self.deltaTheta).sin_().sub_(self.sin).mul_(
            self.wheel_base
        ).div_(self.sin2beta)

        self.deltaY.copy_(pose[:, 2]).add_(self.deltaTheta).cos_().neg_().add_(
            self.cos
        ).mul_(self.wheel_base).div_(self.sin2beta)

        nextpos = self.dtype(self.K, 3)
        nextpos.copy_(pose)
        nextpos[:, 0].add_(self.deltaX)
        nextpos[:, 1].add_(self.deltaY)
        nextpos[:, 2].add_(self.deltaTheta)

        return nextpos
