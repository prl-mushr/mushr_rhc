# Copyright (c) 2019, The Personal Robotics Lab, The MuSHR Team, The Contributors of MuSHR
# License: BSD 3-Clause. See LICENSE.md file in root directory.

import torch
import rospy
import tf
import numpy as np
from visualization_msgs.msg import Marker

class Waypoints:
    NPOS = 3  # x, y, theta

    def __init__(self, params, logger, dtype, map, world_rep):
        self.params = params
        self.logger = logger
        self.dtype = dtype
        self.map = map

        self.world_rep = world_rep

        self.lookahead_publisher = rospy.Publisher(
            "path_points", Marker, queue_size=1
        )

        self.viz_rollouts = self.params.get_bool("debug/flag/viz_rollouts", False)
        self.n_viz = self.params.get_int("debug/viz_rollouts/n", -1)
        self.print_stats = self.params.get_bool("debug/viz_rollouts/print_stats", False)

        self.reset()

    def reset(self):
        self.T = self.params.get_int("T", default=15)
        self.K = self.params.get_int("K", default=62)
        self.dist_w = self.params.get_float("cost_fn/dist_w", default=1.0)
        self.obs_dist_w = self.params.get_float("cost_fn/obs_dist_w", default=5.0)
        self.smoothing_discount_rate = self.params.get_float(
            "cost_fn/smoothing_discount_rate", default=0.04
        )
        self.bounds_cost = self.params.get_float("cost_fn/bounds_cost", default=100.0)

        self.obs_dist_cooloff = torch.arange(1, self.T + 1).mul_(2).type(self.dtype)

        self.discount = self.dtype(self.T - 1)

        self.discount[:] = 1 + self.smoothing_discount_rate
        self.discount.pow_(torch.arange(0, self.T - 1).type(self.dtype) * -1)
        self.world_rep.reset()

    def apply(self, poses, goal, path, car_pose):
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

        # get all collisions (K, T, tensor)
        collisions = self.world_rep.check_collision_in_map(all_poses).view(
            self.K, self.T
        )
        collision_cost = collisions.sum(dim=1).mul(self.bounds_cost)

        obstacle_distances = self.world_rep.distances(all_poses).view(self.K, self.T)
        obstacle_distances[:].mul_(self.obs_dist_cooloff)

        obs_dist_cost = obstacle_distances[:].sum(dim=1).mul(self.obs_dist_w)

        # reward smoothness by taking the integral over the rate of change in poses,
        # with time-based discounting factor
        smoothness = (
            ((poses[:, 1:, 2] - poses[:, : self.T - 1, 2]))
            .abs()
            .mul(self.discount)
            .sum(dim=1)
        )

        cross_track_error = None

        # calculate lookahead
        distance_lookahead = 2

        # calculate closest index to car position
        diff = np.sqrt(((path[:,0] - car_pose[0]) ** 2) + ((path[:,1] - car_pose[1]) ** 2))
        index = np.argmin(diff)

        # iterate to closest lookahead to distance
        while index < len(path) - 1 and diff[index] < distance_lookahead:
            index += 1

        if abs(diff[index - 1] - distance_lookahead) < abs(diff[index] - distance_lookahead):
            index -= 1

        lookahead = path[index]
        quaternion = tf.transformations.quaternion_from_euler(0, 0, lookahead[2])

        marker = Marker()
        marker.pose.position.x = lookahead[0]
        marker.pose.position.y = lookahead[1]
        marker.pose.position.z = lookahead[2]
        marker.pose.orientation.x = quaternion[0]
        marker.pose.orientation.y = quaternion[1]
        marker.pose.orientation.z = quaternion[2]
        marker.pose.orientation.w = quaternion[3]
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.scale.x = .2
        marker.scale.y = .2
        marker.scale.z = .2
        marker.header.frame_id = "map"

        self.lookahead_publisher.publish(marker)

        # ect = -(x - x_ref)sin(theta_ref) + (y - y_ref)cos(theta_ref)
        # TODO: check axis
        cross_track_error = np.abs(-(poses[:, :, 0] - lookahead[0]) * np.sin(lookahead[2]) + (poses[:, :, 1] - lookahead[1]) * np.cos(lookahead[2]))
        cross_track_error = torch.sum(cross_track_error, dim=1)

        # multiply weights
        cross_track_error *= 1000

        result = collision_cost.add(obs_dist_cost).add(smoothness).add(cross_track_error)
        # result = cross_track_error

        colliding = collision_cost.nonzero()
        result[colliding] = 1000000000

        if self.viz_rollouts:
            import librhc.rosviz as rosviz

            non_colliding = (collision_cost == 0).nonzero()

            if non_colliding.size()[0] > 0:

                def print_n(c, poses, ns, cmap="coolwarm"):
                    _, all_idx = torch.sort(c)

                    n = min(self.n_viz, len(c))
                    idx = all_idx[:n] if n > -1 else all_idx
                    rosviz.viz_paths_cmap(poses[idx], c[idx], ns=ns, cmap=cmap)

                p_non_colliding = poses[non_colliding].squeeze()
                print_n(
                    result[non_colliding].squeeze(), p_non_colliding, ns="final_result"
                )
                print_n(
                    collision_cost[non_colliding].squeeze(),
                    p_non_colliding,
                    ns="collision_cost",
                )
                print_n(
                    obs_dist_cost[non_colliding].squeeze(),
                    p_non_colliding,
                    ns="obstacle_dist_cost",
                )
                print_n(
                    smoothness[non_colliding].squeeze(),
                    p_non_colliding,
                    ns="smoothness",
                )

                if self.print_stats:
                    _, all_sorted_idx = torch.sort(result[non_colliding].squeeze())
                    n = min(self.n_viz, len(all_sorted_idx))
                    idx = all_sorted_idx[:n] if n > -1 else all_sorted_idx

                    print("Final Result")
                    print(result[idx])
                    print("Collision Cost")
                    print(collision_cost[idx])
                    print("Obstacle Distance Cost")
                    print(obs_dist_cost[idx])
                    print("Smoothness")
                    print(smoothness[idx])

        return result
    
    def set_goal(self, goal):
        self.goal = goal
        return True
