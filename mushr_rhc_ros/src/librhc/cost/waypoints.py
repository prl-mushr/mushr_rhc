import torch

class Waypoints:
    NPOS = 3 # x, y, theta

    def __init__(self, params, logger, dtype, world_rep, value_fn):
        self.params = params
        self.logger = logger
        self.dtype = dtype

        self.world_rep = world_rep
        self.value_fn = value_fn

        self.reset()

    def reset(self):
        self.T = self.params.get_int("T", default=15)
        self.K = self.params.get_int("K", default=62)
        self.bounds_cost = self.params.get_float("cost_fn/bounds_cost", default=100.0)

    def apply(self, poses, goal):
        """
        Arguments:
            poses (K, T, 3 tensor): Roll out of T positions
            goal  (3, tensor): Goal position
        Returns:
            (K, tensor) of costs for each path
        """
        assert poses.size() == (self.K, self.T, self.NPOS)
        assert goal.size() == (self.NPOS,)

        all_poses = poses.view(self.K * self.T, self.NPOS)

        # Use the x and y coordinates to compute the distance
        dists_raw = all_poses[:, :2].sub(goal[:2])
        dists = dists_raw.norm(p=2, dim=1)

        collisions = self.world_rep.collisions(all_poses)

        val_fn_poses = poses[:, self.T-1, :]#.clone().cpu().numpy()
        cost2go = self.value_fn.get_value(val_fn_poses)

        costs = torch.add(dists, self.bounds_cost, collisions)

        result = costs.view(self.K, self.T).sum(dim=1)
        result += cost2go.type(self.dtype)

        return  result

