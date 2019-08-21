# Copyright (c) 2019, The Personal Robotics Lab, The MuSHR Team, The Contributors of MuSHR
# License: BSD 3-Clause. See LICENSE.md file in root directory.

import os
from threading import Event

import networkx as nx
import numpy as np
import torch
from scipy.interpolate import interp1d
from sklearn.neighbors import NearestNeighbors

import librhc.utils as utils


def next_prime():
    def is_prime(num):
        "Checks if num is a prime value"
        for i in range(2, int(num ** 0.5) + 1):
            if (num % i) == 0:
                return False
        return True

    prime = 3
    while 1:
        if is_prime(prime):
            yield prime
        prime += 2


def vdc(n, base=2):
    vdc, denom = 0, 1
    while n:
        denom *= base
        n, remainder = divmod(n, base)
        vdc += remainder / float(denom)
    return vdc


def halton_sequence(size, dim):
    seq = []
    primeGen = next_prime()
    next(primeGen)
    for d in range(dim):
        base = next(primeGen)
        seq.append([vdc(i, base) for i in range(size)])
    return seq


class SimpleKNN:
    def __init__(self, params, logger, dtype, map):
        self.params = params
        self.logger = logger
        self.map = map
        self.dtype = dtype
        self.nbrs = None
        self.goal_i = None
        self.goal_event = Event()

        self.perm_region = utils.map.load_permissible_region(self.params, map)
        h, w = self.perm_region.shape

        nhalton = self.params.get_int("value/simpleknn/nhalton", default=3000)
        map_cache = utils.cache.get_cache_map_dir(self.params, self.map)
        halton_pts_file = os.path.join(map_cache, "halton-{}.npy".format(nhalton))
        if os.path.isfile(halton_pts_file):
            self.points = np.load(halton_pts_file)
        else:
            self.points = self._iterative_sample_seq(
                h, w, nhalton, self._halton_sampler
            )
            np.save(halton_pts_file, self.points)

    def _halton_sampler(self, h, w, n):
        # re-sample halton with more points
        seq = halton_sequence(n, 2)
        # get number of points in available area
        return [(int(s[0] * h), int(s[1] * w)) for s in zip(seq[0], seq[1])]

    def _iterative_sample_seq(self, h, w, threshold=300, sampler=None):
        assert sampler is not None

        n = threshold * 5
        inc = threshold * 2

        valid = []
        while len(valid) < threshold:
            valid = []

            all_points = sampler(h, w, n)

            for y, x in all_points:
                # if it's a valid points, append to valid_points
                if self.perm_region[y, x] == 0:
                    valid.append((y, x))
            n += inc
            print("valid points len: " + str(len(valid)))
        return np.array(valid)

    def set_goal(self, goal, n_neighbors=7, k=3):
        """
        Args:
        goal [(3,) tensor] -- Goal in "world" coordinates
        Return:
        [boolean] -- Whether the goal was successfully set.
        """
        assert goal.size() == (3,)
        self.goal_event.clear()

        # Convert "world" goal to "map" coordinates
        goal = goal.unsqueeze(0)
        map_goal = self.dtype(goal.size())
        utils.map.world2map(self.map, goal, out=map_goal)

        # Add goal to points on the map so we can create a single source shortest path from it
        map_goal = np.array([[map_goal[0, 1], map_goal[0, 0]]])
        pts_w_goal = np.concatenate((self.points, map_goal), axis=0)

        self.goal_i = pts_w_goal.shape[0] - 1

        # Create a single source shortest path
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="ball_tree").fit(
            pts_w_goal
        )
        distances, indices = nbrs.kneighbors(pts_w_goal)
        elist = []
        for idx_set in indices:
            starti = idx_set[0]
            start = pts_w_goal[starti]
            for n in idx_set[1:]:
                neigh = pts_w_goal[n]
                dist = self._eval_edge(start, neigh)
                if dist > 0:
                    elist.append((starti, n, dist))

        G = nx.Graph()
        G.add_weighted_edges_from(elist)

        try:
            length_to_goal = nx.single_source_dijkstra_path_length(G, self.goal_i)
        except nx.NodeNotFound:
            # There was no paths to this point
            return False

        self.reachable_pts = pts_w_goal[length_to_goal.keys()]
        self.reachable_nodes = length_to_goal.keys()
        self.reachable_dst = length_to_goal.values()

        self._viz_halton()

        self.nbrs = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(
            self.reachable_pts
        )
        self.goal_event.set()
        return True

    def _viz_halton(self):
        import rospy
        from visualization_msgs.msg import Marker
        from geometry_msgs.msg import Point
        from std_msgs.msg import ColorRGBA

        hp = np.zeros((len(self.reachable_pts), 3))
        hp[:, 0] = self.reachable_pts[:, 1]
        hp[:, 1] = self.reachable_pts[:, 0]
        utils.map.map2worldnp(self.map, hp)

        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = rospy.Time.now()
        m.ns = "hp"
        m.id = 0
        m.type = m.POINTS
        m.action = m.ADD
        m.pose.position.x = 0
        m.pose.position.y = 0
        m.pose.position.z = 0
        m.pose.orientation.x = 0.0
        m.pose.orientation.y = 0.0
        m.pose.orientation.z = 0.0
        m.pose.orientation.w = 1.0
        m.scale.x = 0.1
        m.scale.y = 0.1
        m.scale.z = 0.1
        max_d = np.max(self.reachable_dst)
        for i, pts in enumerate(hp):
            p = Point()
            c = ColorRGBA()
            c.a = 1
            c.g = int(255.0 * self.reachable_dst[i] / max_d)
            p.x, p.y = pts[0], pts[1]
            m.points.append(p)
            m.colors.append(c)

        pub = rospy.Publisher("~markers", Marker, queue_size=100)
        pub.publish(m)

    def _eval_edge(self, src, dst):
        # moves along line between src and dst and samples points
        # in the line,
        # return l2 norm if valid and -1 otherwise
        x = np.array([src[0], dst[0]])
        y = np.array([src[1], dst[1]])
        f = interp1d(x, y)
        xs = np.linspace(src[0], dst[0], num=10, endpoint=True)
        ys = f(xs)
        for x, y in zip(xs, ys):
            if self.perm_region[int(x), int(y)] == 1:
                return -1
        return np.linalg.norm(src - dst)

    def get_value(self, input_poses, resolution=None):
        """
        *NOTE* This function does not take angle of car into account.

        Args:
        input_poses [(K, NPOS) tensor] -- Terminal poses for rollouts to be evaluated in "world" coordinates

        Returns:
        [(K,) tensor] -  Cost to go terminal values
        """

        if not self.goal_event.is_set():
            return torch.zeros(len(input_poses)).type(self.dtype)

        if self.nbrs is None:
            return torch.zeros(len(input_poses)).type(self.dtype)

        input_points = input_poses.clone().cpu().numpy()
        utils.map.world2mapnp(self.map, input_points)

        input_points_corrected = input_points.copy()
        input_points_corrected[:, 0] = input_points[:, 1]
        input_points_corrected[:, 1] = input_points[:, 0]
        distances, indices = self.nbrs.kneighbors(input_points_corrected[:, :2])
        result = np.zeros(len(input_points))

        for i in range(len(input_points)):
            idx_set = indices[i]
            min_len = 10e5
            for j, n in enumerate(idx_set):
                min_len = min(2 * self.reachable_dst[n] + distances[i][j], min_len)
            result[i] = min_len

        return torch.from_numpy(result).type(self.dtype) * self.map.resolution
