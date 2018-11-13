import torch

class Waypoints:
    def __init__(self, params, logger, dtype, world_rep):
        self.params = params
        self.logger = logger
        self.dtype = dtype
        self.T = params.get_int("T", default=15)
        self.K = params.get_int("K", default=62)
        self.bounds_cost = params.get_float("bounds_cost", default=100.0)

        self.world_rep = world_rep

    def apply(self, poses, goal):
        """
        Arguments:
            poses (K, T, 3 tensor): Roll out of T positions
            goal  (3, tensor): Goal position
        """
        assert poses.size() == (self.K, self.T, 3)
        assert goal.size() == (3,)

        all_poses = poses.view(self.K * self.T, 3)

        dist_raw = all_poses[:, :2].sub(goal[:2])

        # Use the x and y coordinates to compute the distance
        dists = dist_raw.norm(p=2, dim=1)

        collisions = self.world_rep.collisions(all_poses)

        costs = torch.add(dists, self.bounds_cost, collisions)

        return costs.view(self.K, self.T).sum(dim=1)

