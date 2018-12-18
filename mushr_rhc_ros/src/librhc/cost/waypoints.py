import librhc.rosviz as rosviz


class Waypoints:
    NPOS = 3  # x, y, theta

    def __init__(self, params, logger, dtype, map, world_rep, value_fn):
        self.params = params
        self.logger = logger
        self.dtype = dtype
        self.map = map

        self.world_rep = world_rep
        self.value_fn = value_fn

        self.reset()

    def reset(self):
        self.T = self.params.get_int("T", default=20)
        self.K = self.params.get_int("K", default=62)
        self.bounds_cost = self.params.get_float("cost_fn/bounds_cost",
                                                 default=1000.0)

        self.world_rep.reset()

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

        # uses all the distances, not just terminal
        # dists_raw = all_poses[:, :2].sub(goal[:2])
        # dists = dists_raw.view(self.K,
        #                        self.T,
        #                        self.NPOS).norm(p=2, dim=1).mul_(10)

        # use terminal distance (K, tensor)
        dists = poses[:, self.T-1, :2].sub(goal[:2]).norm(p=2, dim=1).mul(0.5)
        cost2go = self.value_fn.get_value(poses[:, self.T-1, :]).mul(10)

        # get all collisions (K, T, tensor)
        collisions = self.world_rep.check_collision_in_map(
                        all_poses).view(self.K, self.T)
        # collisions = \
        #        self.world_rep.collisions(
        #           poses[:,self.T-1,:].view(self.K,self.NPOS)).view(self.K, 1)
        obstacle_distances = self.world_rep.distances(
                                all_poses).view(self.K, self.T)

        collision_cost = collisions.sum(dim=1).mul(self.bounds_cost)
        obstacle_dist_cost = obstacle_distances.sum(dim=1)
        # .exponential_().add(1).pow(-1).mul(20)
        result = dists.add(cost2go).add(collision_cost).add(obstacle_dist_cost)

        rosviz.viz_paths_cmap(poses, result, cmap='coolwarm')

        '''
        import sys
        sys.stderr.write("\x1b[2J\x1b[H")
        print "Dists: "
        print str(dists)
        print "Cost2Go: "
        print str(cost2go)
        print "Collisions: "
        print str(collision_cost)
        print "Obstacle Dist Cost: "
        print str(obstacle_dist_cost)
        print "Results: "
        print str(result)
        raw_input("Hit enter:")
        '''

        return result
