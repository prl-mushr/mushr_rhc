import torch


class Waypoints:
    NPOS = 3  # x, y, theta

    def __init__(self, params, logger, dtype, map, world_rep, value_fn):
        self.params = params
        self.logger = logger
        self.dtype = dtype
        self.map = map

        self.world_rep = world_rep
        self.value_fn = value_fn

        self.viz_rollouts = self.params.get_bool("debug/flag/viz_rollouts", False)
        self.n_viz = self.params.get_int("debug/viz_rollouts/n", -1)
        self.print_stats = self.params.get_bool("debug/viz_rollouts/print_stats", False)

        self.reset()

    def reset(self):
        self.T = self.params.get_int("T", default=15)
        self.K = self.params.get_int("K", default=62)
        self.dist_w = self.params.get_float("cost_fn/dist_w", default=1.0)
        self.obs_dist_w = self.params.get_float("cost_fn/obs_dist_w", default=5.0)
        self.cost2go_w = self.params.get_float("cost_fn/cost2go_w", default=1.0)
        self.smoothing_discount_rate = self.params.get_float("cost_fn/smoothing_discount_rate", default=0.04)
        self.bounds_cost = self.params.get_float("cost_fn/bounds_cost",
                                                 default=100.0)

        self.discount = self.dtype(self.T-1)
        self.discount[:] = 1 + self.smoothing_discount_rate
        self.discount.pow_(torch.arange(0, self.T-1).type(self.dtype) * -1)
        self.world_rep.reset()

    def apply(self, poses, goal):
        """
        Args:
        poses [(K, T, 3) tensor] -- Rollout of T positions
        goal  [(3,) tensor]: Goal position in "world" mode

        Returns:
        [(K,) tensor] costs for each K paths
        """
        assert poses.size() == (self.K, self.T, self.NPOS)
        assert goal.size() == (self.NPOS,)

        all_poses = poses.view(self.K * self.T, self.NPOS)

        # use terminal distance (K, tensor)
        cost2go = self.value_fn.get_value(poses[:, self.T-1, :]).mul(self.cost2go_w)

        # get all collisions (K, T, tensor)
        collisions = self.world_rep.check_collision_in_map(all_poses).view(self.K, self.T)
        collision_cost = collisions.sum(dim=1).mul(self.bounds_cost)

        obstacle_distances = self.world_rep.distances(all_poses).view(self.K, self.T)

        obs_dist_cost = obstacle_distances[:].sum(dim=1).mul(self.obs_dist_w)

        # reward smoothness by taking the integral over the rate of change in poses,
        # with time-based discounting factor
        smoothness = ((poses[:, 1:, 2] - poses[:, :self.T-1, 2])).abs().mul(self.discount).sum(dim=1)

        # result = dists.add(cost2go).add(collision_cost).add(obstacle_dist_cost).add(smoothness)
        result = cost2go.add(collision_cost).add(obs_dist_cost).add(smoothness)

        if self.viz_rollouts:
            import librhc.rosviz as rosviz

            def print_n(c, ns, cmap='coolwarm'):
                _, all_idx = torch.sort(c)
                idx = all_idx[:self.n_viz] if self.n_viz > -1 else all_idx
                rosviz.viz_paths_cmap(poses[idx], c[idx], ns=ns, cmap=cmap)
                if self.print_stats:
                    self.logger.info(ns)
                    self.logger.info("Min: " + str(torch.min(c)) +
                                     ", Max: " + str(torch.max(c)) +
                                     ", Avg: " + str(torch.mean(c)))

            print_n(result, ns="final_result")
            print_n(cost2go, ns="cost2go")
            print_n(collision_cost, ns="collision_cost")
            print_n(obs_dist_cost, ns="obstacle_dist_cost")
            print_n(smoothness, ns="smoothness")

        return result
