# Copyright (c) 2019, The Personal Robotics Lab, The MuSHR Team, The Contributors of MuSHR
# License: BSD 3-Clause. See LICENSE.md file in root directory.

import os
from itertools import product

import torch
from scipy.spatial.distance import directed_hausdorff

import librhc.utils as utils


class Dispersion:
    NCTRL = 2
    NPOS = 3
    # Needed so the range doesn't cut out the max delta
    DELTAS_EPSILON = 0.0001

    def __init__(self, params, logger, dtype, motion_model):
        self.logger = logger
        self.params = params
        self.dtype = dtype
        self.motion_model = motion_model

        self.reset()

    def reset(self):
        self.K = self.params.get_int("K", default=62)
        self.T = self.params.get_int("T", default=15)

        # Number of resamples in control space
        branching_factor = self.params.get_int(
            "trajgen/dispersion/branching_factor", default=3
        )
        samples = self.params.get_int("trajgen/dispersion/samples", default=5)

        # Total number of controls in mother set
        N_mother = samples ** branching_factor

        min_delta = self.params.get_float("trajgen/min_delta", default=-0.34)
        max_delta = self.params.get_float("trajgen/max_delta", default=0.34)

        desired_speed = self.params.get_float("trajgen/desired_speed", default=1.0)

        # Sample control space
        step_size = (max_delta - min_delta) / (samples - 1)
        deltas = torch.arange(min_delta, max_delta + self.DELTAS_EPSILON, step_size)

        controls_per_branch = int(self.T / branching_factor)

        assert deltas.size() == (samples,)
        assert self.T % branching_factor == 0

        # TODO: Have a cache system for these trajectories that would get
        # loaded here instead of doing needless computation
        dispersion_dir = utils.cache.get_cache_dir(
            self.params, os.path.join("trajgen", "dispersion")
        )
        dispersion_name = "-".join(
            map(
                str,
                [
                    self.K,
                    self.T,
                    branching_factor,
                    samples,
                    max_delta,
                    min_delta,
                    desired_speed,
                ],
            )
        )
        dispersion_name = dispersion_name + ".pt"
        dispersion_path = os.path.join(dispersion_dir, dispersion_name)

        if not os.path.isdir(dispersion_dir):
            os.mkdirs(dispersion_dir)

        if os.path.isfile(dispersion_path):
            self.logger.debug("Loading dispersion from cache: " + dispersion_path)
            self.ctrls = torch.load(dispersion_path)
        else:
            cartesian_prod = product(*[deltas for i in range(branching_factor)])
            prod = self.dtype(list(cartesian_prod))
            ms_deltas = (
                prod.view(-1, 1).repeat(1, controls_per_branch).view(N_mother, self.T)
            )

            # Add zero control
            zero_ctrl = self.dtype(self.T).zero_()
            zero_idx = -1
            for i, c in enumerate(ms_deltas):
                if torch.equal(c, zero_ctrl):
                    zero_idx = i
            if zero_idx >= 0:
                ms_ctrls = self.dtype(N_mother, self.T, self.NCTRL)
                ms_ctrls[:, :, 0] = desired_speed
                ms_ctrls[:, :, 1] = ms_deltas
            else:
                zero_idx = 0
                ms_ctrls = self.dtype(N_mother + 1, self.T, self.NCTRL)
                ms_ctrls[:, :, 0] = desired_speed
                ms_ctrls[1:, :, 1] = ms_deltas
                ms_ctrls[0, :, 1] = 0

            ms_poses = self._rollout_ms(ms_ctrls)
            self._prune_mother_set(zero_idx, ms_ctrls, ms_poses)
            torch.save(self.ctrls, dispersion_path)

    def _rollout_ms(self, ms_ctrls):
        # rollout the mother set
        k = self.motion_model.K
        self.motion_model.set_k(len(ms_ctrls))
        ms_poses = self.dtype(len(ms_ctrls), self.T, self.NPOS).zero_()
        for t in range(1, self.T):
            cur_x = ms_poses[:, t - 1]
            cur_u = ms_ctrls[:, t - 1]
            ms_poses[:, t] = self.motion_model.apply(cur_x, cur_u)
        self.motion_model.set_k(k)
        return ms_poses

    def _prune_mother_set(self, zero_idx, ms_ctrls, ms_poses):
        visited = {zero_idx: ms_poses[zero_idx]}
        dist_cache = {}

        def hausdorff(a, b):
            return max(directed_hausdorff(a, b)[0], directed_hausdorff(b, a)[0])

        for _ in range(self.K - 1):
            max_i, max_dist = 0, 0
            for rollout in range(len(ms_ctrls)):
                if rollout in visited:
                    continue

                min_dist = 10e10
                for idx, visited_rollout in visited.items():
                    if (idx, rollout) not in dist_cache:
                        d = hausdorff(visited_rollout[:, :2], ms_poses[rollout, :, :2])
                        dist_cache[(idx, rollout)] = d
                        dist_cache[(rollout, idx)] = d
                    min_dist = min(dist_cache[(idx, rollout)], min_dist)

                if min_dist > max_dist:
                    max_i, max_dist = rollout, min_dist

            visited[max_i] = ms_poses[max_i]

        assert len(visited) == self.K
        self.ctrls = self.dtype(self.K, self.T, self.NCTRL)
        self.ctrls.copy_(ms_ctrls[visited.keys()])

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
