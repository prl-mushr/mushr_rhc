import rospy
import torch

class TL:
    # Size of control vector
    NCTRL = 2

    def __init__(self, params, logger, dtype):
        self.K = params.get_int("K", default=62)
        self.T = params.get_int("T", default=15)
	self.logger = logger
        self.dtype = dtype

        self.desired_speed = params.get_float("ctrl_gen/desired_speed", default=1.0)
        self.min_delta = params.get_float("ctrl_gen/min_delta", default=-0.34)
        self.max_delta = params.get_float("ctrl_gen/max_delta", default=0.34)

        step_size = (self.max_delta - self.min_delta) / (self.K - 1)
        self.deltas = torch.arange(
            self.min_delta,
            self.max_delta + step_size,
            step_size)

        # The controls for TL are precomputed, and don't change
        self.ctrls = self.dtype(self.K, self.NCTRL)
        self.ctrls[:, 0] = self.desired_speed
        self.ctrls[:, 1] = self.deltas

    def get_control_trajectories(self):
        '''
          Returns (T, K, NCTRL) vector of controls
            ([:, :, 0] is the desired speed, [:, :, 1] is the control delta)
        '''
        return self.ctrls

    def generate_control(self, controls, costs):
        '''
        Inputs
            controls (K, T, NCTRL tensor): Returned by get_controls
            costs (K, 1) cost to take a path

        Retursn
            (T, NCTRL tensor) the lowest cost path
        '''
        _, idx = torch.min(costs)
        return controls[idx]
