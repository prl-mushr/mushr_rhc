class Waypoints:
    def __init__(self, params, world_rep):
        self.T = params.get_int("T")
        self.K = params.get_int("K")
        self.bounds_cost = params.get_float("bounds_cost")

        self.world_rep = world_rep

    def apply(self, poses, goal):
        """
        Arguments:
            poses (K, T, 3 tensor): Roll out of T positions
            goal  (3, tensor): Goal position
        """
        all_poses = poses.view(self.K * self.T)

        dist_raw = self.all_poses.sub(goal)

        # Use the x and y coordinates to compute the distance
        dists = dist_raw.norm(p=2, dim=1, keepdim=True) # (K * T, 1 tensor)

        collisions = self.world_rep.collisions(all_poses)

        cost = dists.add(value=self.bounds_cost, other=collisions)

        return costs.view(self.K, self.T).sum(dim=1)

