import torch
from scipy.signal import savgol_filter


class MXPI:
    # Size of control vector
    NCTRL = 2

    def __init__(self, params, logger, dtype):
        self.logger = logger
        self.params = params
        self.dtype = dtype

        self.reset()

    def reset(self):
        self.K = self.params.get_int("K", default=62)
        self.T = self.params.get_int("T", default=15)

        self.min_delta = self.params.get_float("trajgen/min_delta", default=-0.34)
        self.max_delta = self.params.get_float("trajgen/max_delta", default=0.34)

        # TODO: determine whether to implement with fixed velocity
        self.desired_speed = self.params.get_float("trajgen/desired_speed", default=1.0)

        sigma_v = self.params.get_float("trajgen/mxpi/sigma_v", default=0.15)
        sigma_delta = self.params.get_float("trajgen/mxpi/sigma_delta", default=0.45)
        self.fixed_vel = self.params.get_bool("trajgen/mxpi/fixed_vel", default=True)
        self._lambda = self.params.get_float("trajgen/mxpi/lambda", default=0.62)
        self.use_savgol = self.params.get_bool("trajgen/mxpi/savgol/use", default=True)
        self.savgol_window = self.params.get_int(
            "trajgen/mxpi/savgol/window", default=7
        )
        self.savgol_poly = self.params.get_int("trajgen/mxpi/savgol/poly", default=6)

        self.noise = self.dtype(self.K, self.T, 2)
        self.sigma = self.dtype(self.K, self.T, self.NCTRL)
        self.sigma[:, :, 0] = sigma_v
        self.sigma[:, :, 1] = sigma_delta

        # The controls for TL are precomputed, and don't change
        self.ctrls = self.dtype(self.K, self.T, self.NCTRL)
        self.ctrls[:, :, 0] = self.desired_speed
        self.ctrls[:, :, 1] = 0.0

        self.canonical_ctrl = self.dtype(self.T, 2).zero_()
        self.canonical_ctrl[:, 0] = self.desired_speed

        self.des_ctrl = self.dtype(self.T, 2).zero_()
        self.des_ctrl[:, 0] = self.desired_speed

    def get_control_trajectories(self, velocity):
        """
          Returns (K, T, NCTRL) vector of controls
            ([:, :, 0] is the desired speed, [:, :, 1] is the control delta)
        """
        self.canonical_ctrl[:-1, :] = self.canonical_ctrl[1:, :]
        # Zero out new control or not?
        # self.canonical_ctrl[-1,:] = 0.0
        self.canonical_ctrl[-1, 0] = 0.0
        torch.normal(0, self.sigma, out=self.noise)
        ctrls_expanded = self.canonical_ctrl.expand(self.K, self.T, 2)

        zero_percent = 0.0
        zidx = int(self.K * zero_percent)

        ctrls_expanded[:zidx, :, :] = 0
        self.ctrls.copy_(ctrls_expanded).add_(self.noise)
        # TODO: set fixed speed here
        self.desired_speed = velocity
        if self.fixed_vel:
            self.ctrls[:, :, 0] = velocity
        self.ctrls[:, :, 1].clamp_(self.min_delta, self.max_delta)
        return self.ctrls

    def generate_control(self, controls, costs):
        """
        Inputs
            controls (K, T, NCTRL tensor): Returned by get_controls
            costs (K,) cost to take a path

        Returns
            (T, NCTRL tensor) the lowest cost path
        """
        """
        Args:
        controls [(K, T, NCTRL) tensor] -- Returned by get_control_trajectories
        costs [(K, 1) tensor] -- Cost to take a path

        Returns:
        [(T, NCTRL) tensor] -- The lowest cost trajectory to take
        """
        assert controls.size() == (self.K, self.T, 2)
        assert costs.size() == (self.K,)

        if not self.fixed_vel:
            vels = (
                torch.sum(
                    -1 * (torch.abs(controls[:, :, 0]) - self.desired_speed), dim=1
                )
                * 0.1
            )
            vels += (
                torch.sum(-1 * (controls[:, :, 0] - self.desired_speed), dim=1) * 0.1
            )
            costs.add_(vels)
        beta, idx = costs.min(0)
        costs -= beta
        costs /= -self._lambda
        costs.exp_()
        eta = torch.sum(costs, -1)
        costs /= eta
        weights = costs.unsqueeze(1).expand(self.K, self.T)
        weighted_v_noise = weights * self.noise[:, :, 0]
        weighted_delta_noise = weights * self.noise[:, :, 1]
        v_noise_sum = torch.sum(weighted_v_noise, dim=0)  # (T,)
        delta_noise_sum = torch.sum(weighted_delta_noise, dim=0)  # (T,)
        if self.use_savgol:
            v_noise_sum = torch.from_numpy(
                savgol_filter(v_noise_sum.numpy(), self.savgol_window, self.savgol_poly)
            )
            delta_noise_sum = torch.from_numpy(
                savgol_filter(
                    delta_noise_sum.numpy(), self.savgol_window, self.savgol_poly
                )
            )

        # TODO: reclamp after addition
        #        self.canonical_ctrl[:,0] += v_noise_sum
        #        self.canonical_ctrl[:,0].clamp_(-self.desired_speed, self.desired_speed)
        # TODO: make speed selection optional -- fixed speed for now
        if self.fixed_vel:
            self.canonical_ctrl[:, 0] = self.desired_speed
        else:
            self.canonical_ctrl[:, 0] += v_noise_sum
            self.canonical_ctrl[:, 0].clamp_(-self.desired_speed, self.desired_speed)
        self.canonical_ctrl[:, 1] += delta_noise_sum
        self.canonical_ctrl[:, 1].clamp_(self.min_delta, self.max_delta)
        return self.canonical_ctrl, None
