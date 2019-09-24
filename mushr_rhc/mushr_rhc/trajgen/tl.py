# Copyright (c) 2019, The Personal Robotics Lab, The MuSHR Team, The Contributors of MuSHR
# License: BSD 3-Clause. See LICENSE.md file in root directory.

import torch


class TL:
    # Size of control vector
    NCTRL = 2

    def __init__(self, params, logger, dtype, _model):
        self.logger = logger
        self.params = params
        self.dtype = dtype

        self.reset()

    def reset(self):
        self.K = self.params.get_int("K", default=62)
        self.T = self.params.get_int("T", default=15)

        min_delta = self.params.get_float("trajgen/min_delta", default=-0.34)
        max_delta = self.params.get_float("trajgen/max_delta", default=0.34)

        desired_speed = self.params.get_float("trajgen/desired_speed", default=1.0)
        step_size = (max_delta - min_delta) / (self.K - 1)
        deltas = torch.arange(min_delta, max_delta + step_size, step_size)

        # The controls for TL are precomputed, and don't change
        self.ctrls = self.dtype(self.K, self.T, self.NCTRL)
        self.ctrls[:, :, 0] = desired_speed
        for t in range(self.T):
            self.ctrls[:, t, 1] = deltas

    def get_control_trajectories(self, velocity):
        """
        Returns:
        [(K, T, NCTRL) tensor] -- of controls
            ([:, :, 0] is the desired speed, [:, :, 1] is the control delta)
        """
        self.ctrls[:, :, 0] = velocity
        return self.ctrls

    def generate_control(self, controls, costs):
        """
        Args:
        controls [(K, T, NCTRL) tensor] -- Returned by get_control_trajectories
        costs [(K, 1) tensor] -- Cost to take a path

        Returns:
        [(T, NCTRL) tensor] -- The lowest cost trajectory to take
        """
        assert controls.size() == (self.K, self.T, 2)
        assert costs.size() == (self.K,)
        _, idx = torch.min(costs, 0)
        return controls[idx], idx
