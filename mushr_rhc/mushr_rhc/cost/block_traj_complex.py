# Copyright (c) 2019, The Personal Robotics Lab, The MuSHR Team, The Contributors of MuSHR
# License: BSD 3-Clause. See LICENSE.md file in root directory.

import torch
import mushr_rhc.utils as utils
import threading
import math

# send marker
import rospy
import tf.transformations
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Quaternion


def a2q(a):
    return Quaternion(*tf.transformations.quaternion_from_euler(0, 0, a))
# end send


class BlockRefTrajectoryComplex:
    def __init__(
        self, params, logger, dtype, map, world_rep, value_fn, viz_rollouts_fn
    ):
        self.params = params
        self.logger = logger
        self.dtype = dtype
        self.map = map

        self.world_rep = world_rep
        self.value_fn = value_fn

        self.viz_rollouts_fn = viz_rollouts_fn

        self.DEBUG = rospy.Publisher("~current_marker", Marker, queue_size=1)
        self.BACK_PT = rospy.Publisher("~backpoint", MarkerArray, queue_size=1)
        self.BLOCKARRAY = rospy.Publisher("~block_viz", MarkerArray, queue_size=1)

        self.reset()

    def reset(self):
        self.T = self.params.get_int("T", default=15)
        self.K = self.params.get_int("K", default=62)
        self.NPOS = self.params.get_int("npos", default=3)

        time_horizon = utils.get_time_horizon(self.params)
        self.dt = time_horizon / self.T

        self.dist_horizon = utils.get_distance_horizon(self.params)
        self.traj_lock = threading.RLock()
        with self.traj_lock:
            self.goal = None
        self.goal_threshold = self.params.get_float("xy_threshold", default=0.2)

        horizon = utils.get_distance_horizon(self.params)
        self.waypoint_lookahead = 1.0  # 0.2  # self.dist_horizon
        self.waypoint_idx_lookahead = int(math.ceil(horizon / 0.5)) + 3  # 3

        self.dist_w = self.params.get_float("cost_fn/dist_w", default=1.0)

        self.a_diff_w = 1.0
        self.block_car_dist_w = 1.0
        self.block_car_dist_shift = 2.5

        self.contact_mode = True

        self.world_rep.reset()

    def apply(self, poses, _cov=None):
        """
        Args:
        poses [(K, T, 3) tensor] -- Rollout of T positions
        goal  [(3,) tensor]: Goal position in "world" mode

        Returns:
        [(K,) tensor] costs for each K paths
        """
        assert poses.size() == (self.K, self.T, self.NPOS)

        with self.traj_lock:
            waypoint = self.get_waypoint(poses[0, 0, 3:5])

        # dist = torch.norm(poses[:, self.T - 1, 3:5] - waypoint[:2], dim=1).mul_(self.dist_w)

        # all_dist = torch.norm(poses[:, :, 3:5] - waypoint[:2], dim=2)
        # traj_dists = torch.sum(all_dist, dim=1).div(self.T)

        ##
        # from block traj
        ##

        s_block_goal_vec = waypoint[:2] - poses[0, 0, 3:5]  # (2, )
        s_block_goal_dist = s_block_goal_vec.pow(2).sum(dim=0).pow_(0.5)

        s_block_car_vec = poses[0, 0, :2] - poses[0, 0, 3:5]  # (2,)
        s_block_car_dist = s_block_car_vec.pow(2).sum(dim=0).pow_(0.5)

        final_idx = min(self.T - 1, int(s_block_goal_dist / self.dt))

        car_goal_angle = s_block_goal_vec.dot(s_block_car_vec)
        car_goal_angle = car_goal_angle.div_(
            torch.norm(s_block_goal_vec) * torch.norm(s_block_car_vec)
        )
        car_goal_angle = car_goal_angle.acos_()
        if car_goal_angle < 0:
            car_goal_angle += 2 * math.pi

        # step 1 if the car is in between block and goal, get to not there.
        # step 2 one past there, turn into the block such that it can be moved straight to the goal
        # step 3, once close enough to the block, use block distance to goal as cost

        # vector from goal to goal and car respectively (from the final point of the rollout)
        f_block_goal_vec = waypoint[:2] - poses[:, final_idx, 3:5]  # (K, 2)
        f_block_car_vec = poses[:, final_idx, :2] - poses[:, final_idx, 3:5]  # (K, 2)

        f_block_car_dist = f_block_car_vec.pow(2).sum(dim=1).pow_(0.5)  # (K,)
        f_block_goal_dist = f_block_goal_vec.pow(2).sum(dim=1).pow_(0.5)  # (K,)

        angles = f_block_goal_vec.mul(f_block_car_vec).sum(dim=1)
        angles.div_(f_block_goal_dist).div_(f_block_car_dist).acos_()
        angles[angles < 0] += 2 * math.pi

        backward = False

        if not (
            s_block_car_dist < 3.0
            and 3.0 / 4.0 * math.pi <= car_goal_angle
            and car_goal_angle <= 5.0 / 4.0 * math.pi
        ):
            self.contact_mode = False

        if not (
            self.contact_mode
            # 3.0 / 4.0 * math.pi <= car_goal_angle
            # and car_goal_angle <= 5.0 / 4.0 * math.pi
        ):
            # block_pos - 1.5 * unit(block->goal)
            L = 2.2
            Theta = math.pi / 10
            phi = torch.asin(s_block_goal_vec[1] / torch.norm(s_block_goal_vec))
            block = poses[0, 0, 3:5]

            p1 = block + L * self.dtype([torch.cos(math.pi + phi - Theta),
                                         torch.sin(math.pi + phi - Theta)])
            p2 = block + L * self.dtype([torch.cos(math.pi + phi + Theta),
                                         torch.sin(math.pi + phi + Theta)])
            p3 = block + L * self.dtype([torch.cos(math.pi + phi),
                                         torch.sin(math.pi + phi)])

            distf1 = torch.norm(p1 - poses[:, self.T - 1, :2], dim=1)
            distf2 = torch.norm(p2 - poses[:, self.T - 1, :2], dim=1)

            dists1 = torch.norm(p1 - poses[:, 0, :2])
            dists2 = torch.norm(p2 - poses[:, 0, :2])

            dist_f = torch.min(distf1, distf2)
            dist_s = torch.min(dists1, dists2)

            if dist_s < 0.5:
                self.contact_mode = True

            ma = MarkerArray()
            for i, p in enumerate([p1, p2, p3]):
                m = Marker()
                m.header.frame_id = "map"
                m.id = i
                m.pose.position.x = p[0]
                m.pose.position.y = p[1]
                m.pose.orientation = a2q(0)
                m.scale.x = 0.1
                m.scale.y = 0.1
                m.scale.z = 0.05
                m.color.r = 1.0
                m.color.a = 1.0
                ma.markers.append(m)
            self.BACK_PT.publish(ma)

            result = dist_f
            backward = True
        elif not (
            s_block_car_dist < 0.7
            # and (
            #     car_goal_angle <= 1.0 / 7.0 * math.pi
            #     or car_goal_angle >= 13.0 / 7.0 * math.pi
            # )
        ):
            # want trajectory that points in the same direction as the block to the goal
            # AND as close to the block as possible (requires first objective tho)

            dist_cost = (
                (poses[:, final_idx, :2] - poses[:, final_idx, 3:5]).pow(2).sum(dim=1)
            )

            a_diff = (angles - math.pi).abs_().mul_(self.a_diff_w)
            print a_diff
            a_diff[a_diff < 0.1] = 0.0

            # TODO: CAN ADD COST TO GO LATER ON MORE COMPLEX MAPS
            # cost2go = self.value_fn.get_value(poses[:, final_idx, 3:]).mul(
            #     self.cost2go_w
            # )
            result = dist_cost.add(a_diff)
            backward = False
        else:
            all_dist = torch.norm(poses[:, :, 3:5] - waypoint[:2], dim=2)
            traj_dists = torch.sum(all_dist, dim=1).div(self.T)

            result = traj_dists.add(f_block_goal_dist)
            backward = False

        ## end block_push
        # send marker
        m = Marker()
        m.header.frame_id = "map"
        m.id = 0
        m.pose.position.x = waypoint[0]
        m.pose.position.y = waypoint[1]
        m.pose.orientation = a2q(waypoint[2])
        m.scale.x = 0.1
        m.scale.y = 0.1
        m.scale.z = 0.05
        m.color.r = 1.0
        m.color.a = 1.0
        self.DEBUG.publish(m)
        # end send

        if self.viz_rollouts_fn:
            self.viz_rollouts_fn(
                result, poses  # , traj_dists=traj_dists,  # offset=offset
            )

            ma = MarkerArray()
            for i, p in enumerate(poses[:, self.T - 1, 3:6]):
                m = Marker()
                m.header.frame_id = "map"
                m.id = i
                m.type = 1
                m.pose.position.x = p[0]
                m.pose.position.y = p[1]
                m.pose.orientation = a2q(p[2])
                m.scale.x = 0.1
                m.scale.y = 0.1
                m.scale.z = 0.05
                m.color.r = float(i) / self.K
                m.color.a = 1.0
                ma.markers.append(m)
            self.BLOCKARRAY.publish(ma)

        return result, backward

    def get_waypoint(self, state):
        dists = torch.norm(self.traj[:, :2] - state[:2], dim=1)
        idx = dists.argmin()
        idx += self.waypoint_idx_lookahead
        idx = min(idx, len(self.traj) - 1)
        return self.traj[idx]

    def set_trajectory(self, traj):
        """
        Args:
        goal [(3,) tensor] -- Goal in "world" coordinates
        """
        with self.traj_lock:
            self.traj = traj
            return True
            # return self.value_fn.set_goal(traj[-1])

    def dist_to_goal(self, state):
        with self.traj_lock:
            if self.traj is None:
                return False
            return self.traj[-1, :2].dist(state[3:5])

    def at_goal(self, state):
        """
        Args:
        state [(3,) tensor] -- Current position in "world" coordinates
        """
        with self.traj_lock:
            if self.traj is None:
                return False
            return self.dist_to_goal(state) < self.goal_threshold

    def get_desired_speed(self, desired_speed, state):
        return min(
            desired_speed,
            desired_speed * (self.dist_to_goal(state) / (self.dist_horizon)),
        )
