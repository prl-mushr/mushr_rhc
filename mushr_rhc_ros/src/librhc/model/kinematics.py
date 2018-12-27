class Kinematics:
    EPSILON = 1e-12
    NPOS = 3

    def __init__(self, params, logger, dtype):
        # init any datastructures needed
        self.logger = logger
        self.params = params
        self.dtype  = dtype

        self.reset()

    def reset(self):
        self.K          = self.params.get_int("K", default=62)
        self.wheel_base = self.params.get_float("model/wheel_base", default=0.33)
        self.dt         = self.params.get_float("model/dt", default=0.1)

        self.sin2beta   = self.dtype(self.K)
        self.deltaTheta = self.dtype(self.K)
        self.deltaX     = self.dtype(self.K)
        self.deltaY     = self.dtype(self.K)
        self.sin        = self.dtype(self.K)
        self.cos        = self.dtype(self.K)


    def apply(self, pose, ctrl):
        '''
        Args:
          pose (K, NPOS tensor): The current position
          ctrl (K, NCTRL tensor): Control to apply to the current position
        Return:
          (K, NCTRL tensor) The next position given the current control
        '''
        assert pose.size() == (self.K, 3)
        assert ctrl.size() == (self.K, 2)

        self.sin2beta.copy_(ctrl[:, 1]).tan_().mul_(0.5).atan_().mul_(2.0).sin_().add_(self.EPSILON)

        self.deltaTheta.copy_(ctrl[:, 0]).div_(self.wheel_base).mul_(self.sin2beta).mul_(self.dt)

        self.sin.copy_(pose[:, 2]).sin_()
        self.cos.copy_(pose[:, 2]).cos_()

        self.deltaX.copy_(pose[:, 2]).add_(self.deltaTheta).sin_().sub_(self.sin).mul_(self.wheel_base).div_(self.sin2beta)

        self.deltaY.copy_(pose[:, 2]).add_(self.deltaTheta).cos_().neg_().add_(self.cos).mul_(self.wheel_base).div_(self.sin2beta)

        nextpos = self.dtype(self.K, 3)
        nextpos.copy_(pose)
        nextpos[:, 0].add_(self.deltaX)
        nextpos[:, 1].add_(self.deltaY)
        nextpos[:, 2].add_(self.deltaTheta)

        return nextpos
