from itertools import product
import torch
from scipy.spatial.distance import directed_hausdorff


class Dispersion:
    NCTRL = 2

    def __init__(self, params, logger, dtype):
        self.logger = logger
        self.params = params
        self.dtype = dtype

        self.reset()

    def reset(self):
        self.K = self.params.get_int('K', default=62)
        self.T = self.params.get_int('T', default=62)
        dt = self.params.get_int('trajgen/dt', default=0.1)

        # Number of seconds lookahead
        time_horizon = self.params.get_int('trajgen/time_horizon', default=3)

        # Number of resamples in control space
        branching_factor = self.params.get_int('trajgen/branching_factor', default=3)
        samples = self.params.get_int('trajgen/samples', default=5)

        # Total number of controls in mother set
        # +1 for zero controls
        N_mother = samples ** branching_factor + 1

        min_delta = self.params.get_float("traj_gen/min_delta", default=-0.34)
        max_delta = self.params.get_float("traj_gen/max_delta", default=0.34)

        desired_speed = self.params.get_float('trajgen/desired_speed',
                                              default=1.0)
        # Sample control space

        step_size = (max_delta - min_delta) / (samples - 1)
        deltas = torch.arange(min_delta, max_delta + step_size, step_size)

        # Numbeor of steps = time_horizon / dt
        assert self.T == time_horizon / dt
        T = int(time_horizon / dt)
        controls_per_branch = int(float(T) / branching_factor)

        cartesian_prod = product(*[deltas for i in range(branching_factor)])
        prod = torch.Tensor(list(cartesian_prod))
        ms_deltas = prod.view(-1, 1).repeat(1, controls_per_branch).view(N_mother, T)

        ms_ctrls = self.dtype(N_mother, T, 2)
        ms_ctrls[:, :, 0] = desired_speed
        ms_ctrls[1:, :, 1] = ms_deltas
        ms_ctrls[0, :, 1] = 0

        # rollout the mother set
        model = Kinematics(N_mother, dt, dtype)
        ms_poses = torch.zeros(N_mother, T, 3)
        for t in range(1, T):
            cur_x = ms_poses[:, t-1]
            ms_poses[:, t] = model.apply(cur_x, ms_ctrls[:, t - 1])

        self.prune_mother_set(ms_ctrls, ms_poses)

    def prune_mother_set(self, ms_ctrls, ms_poses):
        visited = {0: ms_poses[0]}
        dist_cache = {}

        def hausdorff(a, b):
            return max(directed_hausdorff(a, b), directed_hausdorff(b, a))

        for i in range(self.K):
            max_i, max_dist = 0, 0
            for rollout in range(len(ms_ctrls)):
                if rollout in visited:
                    continue

                min_dist = 10e10
                for idx, visited_rollout in visited.items():
                    if (idx, rollout) not in dist_cache:
                        d = hausdorff(visited_rollout[:, :2],
                                      ms_poses[rollout, :, :2])
                        dist_cache[(idx, rollout)] = d
                        dist_cache[(rollout, idx)] = d
                    min_dist = min(dist_cache[(idx, rollout)], min_dist)

                dist = min_dist
                if dist > max_dist:
                    max_i, max_dist = rollout, dist

            visited[max_i] = ms_poses[max_i]

        self.ctrls = self.dtype(len(visited), self.T, self.NCTRL)
        self.ctrls[:, :, :].copy_(ms_ctrls[visited.keys()])

    def get_control_trajectories(self):
        '''
          Returns (K, T, NCTRL) vector of controls
            ([:, :, 0] is the desired speed, [:, :, 1] is the control delta)
        '''
        return self.ctrls

    def generate_control(self, controls, costs):
        '''
        Inputs
            controls (K, T, NCTRL tensor): Returned by get_controls
            costs (K, 1) cost to take a path

        Returns
            (T, NCTRL tensor) the lowest cost path
        '''
        assert controls.size() == (self.K, self.T, 2)
        assert costs.size() == (self.K,)
        _, idx = torch.min(costs, 0)
        return controls[idx]
