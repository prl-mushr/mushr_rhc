# Copyright (c) 2019, The Personal Robotics Lab, The MuSHR Team, The Contributors of MuSHR
# License: BSD 3-Clause. See LICENSE.md file in root directory.

import librhc.utils as utils


class Kinematics:
    EPSILON = 1e-5
    NCTRL = 2
    NPOS = 3

    def __init__(self, params, logger, dtype):
        self.logger = logger
        self.params = params
        self.dtype = dtype

        self.reset()

    def reset(self):
        self.set_k(self.params.get_int("K", default=62))

    def set_k(self, k):
        """
        In some instances the internal buffer size needs to be changed. This easily facilitates this change

        Args:
        k [int] -- Number of rollouts
        """
        self.K = k
        self.wheel_base = self.params.get_float("model/wheel_base", default=0.29)

        time_horizon = utils.get_time_horizon(self.params)
        T = self.params.get_int("T", default=15)
        self.dt = time_horizon / T

        self.sin2beta = self.dtype(self.K)
        self.beta = self.dtype(self.K)
        self.deltaTheta = self.dtype(self.K)
        self.deltaX = self.dtype(self.K)
        self.deltaY = self.dtype(self.K)
        self.sin = self.dtype(self.K)
        self.cos = self.dtype(self.K)

    def apply(self, pose, ctrl):
        """
        Args:
            pose [(K, NPOS) tensor] -- The current position in "world" coordinates
            ctrl [(K, NCTRL) tensor] -- Control to apply to the current position
        Return:
            [(K, NPOS) tensor] The next position given the current control
        """
        assert pose.size() == (self.K, self.NPOS)
        assert ctrl.size() == (self.K, self.NCTRL)

        self.beta.copy_(ctrl[:,1]).tan_().mul_(0.5).atan_()
        self.sin2beta.copy_(self.beta).sin_().mul_(2.0)

        self.deltaTheta.copy_(ctrl[:, 0]).div_(self.wheel_base).mul_(
            self.sin2beta
        ).mul_(self.dt)

        self.sin.copy_(pose[:, 2]).add_(self.beta).sin_()
        self.cos.copy_(pose[:, 2]).add_(self.beta).cos_()

        self.deltaX.copy_(pose[:, 2]).add_(self.deltaTheta).add_(
                self.beta).sin_().sub_(
                        self.sin).mul_(self.wheel_base).div_(self.sin2beta)

        self.deltaY.copy_(pose[:, 2]).add_(
                self.deltaTheta).add_(self.beta).cos_().neg_().add_(
                        self.cos).mul_(self.wheel_base).div_(self.sin2beta)

        nextpos = self.dtype(self.K, 3)
        nextpos.copy_(pose)
        nextpos[:, 0].add_(self.deltaX)
        nextpos[:, 1].add_(self.deltaY)
        nextpos[:, 2].add_(self.deltaTheta)

        # If straight
        nextpos[ctrl[:, 1] < 0.01, 0 ].copy_(pose[ctrl[:, 1] < 0.01, 1]).cos_().mul(ctrl[ctrl[:, 1] < 0.01, 0]) 
        nextpos[ctrl[:, 1] < 0.01, 1 ].copy_(pose[ctrl[:, 1] < 0.01, 1]).sin_().mul(ctrl[ctrl[:, 1] < 0.01, 0]) 
        nextpos[ctrl[:, 1] < 0.01, 2 ].copy_(pose[ctrl[:, 1] < 0.01, 1]) 

        return nextpos
