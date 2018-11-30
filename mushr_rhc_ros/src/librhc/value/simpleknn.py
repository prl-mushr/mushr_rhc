import numpy as np
import torch
import networkx as nx
from scipy.interpolate import interp1d
from sklearn.neighbors import NearestNeighbors
import librhc.utils as utils
from threading import Lock

def next_prime():
    def is_prime(num):
        "Checks if num is a prime value"
        for i in range(2,int(num**0.5)+1):
            if(num % i)==0: return False
        return True

    prime = 3
    while(1):
        if is_prime(prime):
            yield prime
        prime += 2

def vdc(n, base=2):
    vdc, denom = 0, 1
    while n:
        denom *= base
        n, remainder = divmod(n, base)
        vdc += remainder/float(denom)
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
        """
            permissible_region (width x height tensor)
        """
        self.params = params
        self.logger = logger
        self.map = map
        self.dtype = dtype
        self.nbrs = None
        self.goal_i = None
        self.state_lock = Lock()

        self.perm_region = utils.load_permissible_region(self.params, map)
        h, w = self.perm_region.shape

        self.halton_points = self.iterative_sample_halton_pts(h, w)

    def iterative_sample_halton_pts(self, h, w, threshold=300):
        n = threshold * 5
        inc = threshold * 2

        valid = []
        while len(valid) < threshold:
            valid = []

            # re-sample halton with more points
            seq = halton_sequence(n, 2)
            # get number of points in available area
            all_points = [(int(s[0] * h), int(s[1] * w)) for s in zip(seq[0], seq[1])]

            for y, x in all_points:
                # if it's a valid points, append to valid_points
                if self.perm_region[y, x] == 0:
                    valid.append((y,x))
            n += inc
            print "valid points len: " + str(len(valid))
        return np.array(valid)

    def set_goal(self, goal, n_neighbors=7, k=2):
        self.state_lock.acquire()
        # Add goal to self.halton_points
        assert goal.size() == (3,)
        print "goal set in value function " + str(goal)

        goal = goal.unsqueeze(0)
        map_goal = self.dtype(goal.size())
        utils.world2map(self.map, goal, out=map_goal)

        map_goal = np.array([[map_goal[0,1], map_goal[0,0]]])
        self.halton_points = np.concatenate((self.halton_points, map_goal), axis=0)
        self.goal_i = self.halton_points.shape[0]-1

        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(self.halton_points)
        distances, indices = nbrs.kneighbors(self.halton_points)
        elist = []
        for idx_set in indices:
            starti = idx_set[0]
            start = self.halton_points[starti]
            for n in idx_set[1:]:
                neigh = self.halton_points[n]
                dist = self.eval_edge(start, neigh)
                if dist > 0:
                    elist.append((starti,n,dist))

        G = nx.Graph()
        G.add_weighted_edges_from(elist)

        length_to_goal = nx.single_source_dijkstra_path_length(G, self.goal_i)

        self.reachable_pts   = self.halton_points[length_to_goal.keys()]
        self.reachable_nodes = length_to_goal.keys()
        self.reachable_dst   = length_to_goal.values()

        self.viz_halton()

        self.nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(self.reachable_pts)
        self.state_lock.release()
        print "Complete setting goal"

    def viz_halton(self):
        import rospy
        from visualization_msgs.msg import Marker
        from geometry_msgs.msg import Point

        hp = np.zeros((len(self.halton_points), 3))
        hp[:, 0] = self.halton_points[:,1]
        hp[:, 1] = self.halton_points[:,0]
        utils.map2worldnp(self.map, hp)

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
        m.scale.x = 1.0
        m.scale.y = 1.0
        m.scale.z = 1.0
        m.color.g = 1.0
        m.color.a = 1.0
        for pts in hp:
            p = Point()
            p.x, p.y = pts[0], pts[1]
            m.points.append(p)

        pub = rospy.Publisher("/haltonpts", Marker, queue_size=100)
        pub.publish(m)
        print "published?"

    def eval_edge(self, src, dst):
        # moves along line between src and dst and samples points
        # in the line,
        # return l2 norm if valid and -1 otherwise
        x = np.array([src[0], dst[0]])
        y = np.array([src[1], dst[1]])
        f = interp1d(x, y)
        xs = np.linspace(src[0], dst[0], num=10, endpoint=True)
        ys = f(xs)
        for x,y in zip(xs,ys):
            if self.perm_region[int(x), int(y)] == 1:
                return -1
        return np.linalg.norm(src-dst)

    def get_value(self, input_poses, resolution=None):
        """
        Arguments:
            input_poses (K, NPOS tensor) Terminal poses for rollouts to be evaluated
        Returns:
            values (K, tensor) Cost to go terminal values

        *Note* This function does not take theta_i into account.
        """

        self.state_lock.acquire()
        if self.goal_i is None:
            print "no goal"
            return torch.zeros(len(input_poses)).type(self.dtype)

        if self.nbrs is None:
            print "nbrs none"
            return torch.zeros(len(input_poses)).type(self.dtype)

        input_points = input_poses.clone().cpu().numpy()
        utils.world2mapnp(self.map, input_points)

        distances, indices = self.nbrs.kneighbors(input_points[:, :2])
        result = np.zeros(len(input_points))
        for i in range(len(input_points)):
            idx_set = indices[i]
            min_len = 10e10
            for j, n in enumerate(idx_set):
                min_len = min(self.reachable_dst[n]+distances[i][j], min_len)
            result[i] = min_len

        self.state_lock.release()
        return torch.from_numpy(result).type(self.dtype) * self.map.resolution
