import mushr_rhc
import mushr_rhc.utils
import torch
import pickle

# import scipy.spatial


class NearestNeighbor:
    EPSILON = 1e-5
    NCTRL = 2

    def __init__(self, params, logger, dtype):
        self.logger = logger
        self.params = params
        self.dtype = dtype

        self.reset()

    def reset(self):
        self.K = self.params.get_int("K", default=62)
        self.T = self.params.get_int("T", default=21)
        self.NPOS = self.params.get_int("npos", default=3)
        self.wheel_base = self.params.get_float("model/wheel_base", default=0.29)

        self.sin2beta = self.dtype(self.K)
        self.deltaTheta = self.dtype(self.K)
        self.deltaX = self.dtype(self.K)
        self.deltaY = self.dtype(self.K)
        self.sin = self.dtype(self.K)
        self.cos = self.dtype(self.K)

        self.diffs = self.dtype(self.K, 5)

        time_horizon = mushr_rhc.utils.get_time_horizon(self.params)
        self.dt = time_horizon / self.T

        with open(self.params.get_str("pushes_start_file"), "rb") as f:
            self.pushes_start = torch.from_numpy(pickle.load(f)).type(self.dtype)

        with open(self.params.get_str("pushes_file"), "rb") as f:
            self.pushes = torch.from_numpy(pickle.load(f)).type(self.dtype)

        self.pushes_min, self.pushes_max = (
            torch.min(self.pushes_start.T, dim=1),
            torch.max(self.pushes_start.T, dim=1),
        )
        self.pushes_min = self.pushes_min.values
        self.pushes_max = self.pushes_max.values
        self.pushes_range = self.pushes_max - self.pushes_min
        self.pushes_range[3] = self.EPSILON  # there is only one velocity

        self.pushes_start_norm = (
            self.pushes_start - self.pushes_min
        ) / self.pushes_range
        print(self.pushes_start_norm)

        # self.hull = scipy.spatial.ConvexHull(self.pushes_start_norm[:, [0, 1, 2]])
        # self.hull_equations = map(lambda x: torch.from_numpy(x).type(torch.FloatTensor), self.hull.equations)

    def rollout(self, state, trajs, rollouts):
        rollouts.zero_()

        # For each K trial, the first position is at the current position
        rollouts[:, 0] = state.expand_as(rollouts[:, 0])

        for t in range(1, self.T):
            cur_x = rollouts[:, t - 1]
            rollouts[:, t] = self.apply(cur_x, trajs[:, t - 1])

    # def point_in_contact_hull(self, point, tolerance=1e-12):
    #     return all((torch.dot(eq[:-1], point[[0, 1, 2]]) + eq[-1] <= tolerance) for eq in self.hull_equations)

    # def in_contact_hull(self, points):
    #     return torch.tensor(map(self.point_in_contact_hull, points))

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

        nextpos = self.dtype(self.K, self.NPOS)
        nextpos.copy_(pose)
        nextpos[:, 0].add_(self.deltaX)
        nextpos[:, 1].add_(self.deltaY)
        nextpos[:, 2].add_(self.deltaTheta)

        self.diffs[:, :3] = pose[:, 3:] - pose[:, :3] - self.pushes_min[:3]
        self.diffs[:, 3:] = ctrl - self.pushes_min[3:]
        self.diffs.div_(self.pushes_range)

        min_dist = self.dtype(pose.shape[0])
        displacement = self.dtype(pose.shape[0], 3)

        # in_hull = self.in_contact_hull(self.diffs)
        # print in_hull

        for i, p in enumerate(self.diffs):
            min_dist[i], idx = torch.min(
                torch.norm(self.pushes_start_norm - p, dim=1), 0
            )
            displacement[i] = self.pushes[idx]

        i = min_dist < 0.5
        # i = in_hull
        # print "min_dist", min_dist
        # print "displacement", displacement[i]

        if torch.any(i):
            nextpos[i, 3] = nextpos[i, 3].add(displacement[i, 0])
            nextpos[i, 4] = nextpos[i, 4].add(displacement[i, 1])
            nextpos[i, 5] = nextpos[i, 5].add(displacement[i, 2])

        return nextpos
