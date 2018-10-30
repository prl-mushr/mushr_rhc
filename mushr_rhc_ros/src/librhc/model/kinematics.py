class Kinematics:
  EPSILON = 1e-13

  def __init__(self, params):
    '''
    Args:
      K = Number of samples
    '''
    # init any datastructures needed
    self.K = params.get_int("K")
    self.wheel_base = params.get_float("movement_model/wheel_base")
    self.dt = params.get_int("movement_model/dt")

  def apply(self, pos, ctrl):
    '''
    Args:
      pos (K, NPOS tensor): The current position
      ctrl (K, NCTRL tensor): Control to apply to the current position
    Return:
      The next position given the current control
    '''

    # pose (velocity, delta, theta)
    # ctrl (velocity, steering)

    # apply the ctrl to the current position
    sin2beta = ctrl[:, 1].copy().tan_().mul_(0.5).atan_().mul_(2.0).sin_().add_(EPSILON)

    ##
    # v = control[:, 0]
    # delta = control[:, 1]
    theta = pose[:, 2]

    # beta = torch.atan(torch.mul(torch.tan(delta), 0.5))
    # sin2beta = torch.sin(torch.mul(beta, 2.0))
    # for numerical stability
    # sin2beta = self.where(sin2beta == 0, \
    #    torch.mul(torch.ones_like(sin2beta), 1e-10), sin2beta)

    dTheta = ctrl[:, 0].copy().div_(
        self.wheel_base).mul_(sin2beta).mul_(self.dt)

    theta_sin = pose[:, 2].sin()
    theta_cos = pose[:, 2].cos()

    dX=pose[:, 2].copy().add_(dTheta).sin_().sub_(
        theta_sin).mul_(wheel_base).div_(sin2beta)

    # (self.robot_length / self.sin2beta) \
    #    * (torch.sin(theta + self.deltaTheta) - torch.sin(theta))
    dY=pose[:, 2].copy().add_(dTheta).cos_().neg_().add_(
        theta_cos).mul_(wheel_base).div_(sin2beta)

    # deltaY = (self.robot_length / self.sin2beta) \
    # * (-torch.cos(theta + self.deltaTheta) + torch.cos(theta))

    next_pos=self.dtype(K, NPOS)
    next_pos[:, 0].copy_(dX).add_(pos[:, 0])
    next_pos[:, 1].copy_(dY).add_(pos[:, 1])
    next_pos[:, 2].copy_(dTheta).add_(pos[:, 2])

    return next_pos
