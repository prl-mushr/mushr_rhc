# Copyright (c) 2019, The Personal Robotics Lab, The MuSHR Team, The Contributors of MuSHR
# License: BSD 3-Clause. See LICENSE.md file in root directory.

import matplotlib.cm as cm
import matplotlib.colors as colors
import rospy
import torch
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray


class VizPaths:
    def __init__(self):
        self.traj_pub = rospy.Publisher(
            rospy.get_param("~debug/viz_rollouts/topic", "~debug/viz_rollouts"),
            MarkerArray,
            queue_size=100,
        )
        self.n_viz = int(rospy.get_param("debug/viz_rollouts/n", -1))
        self.print_stats = bool(
            rospy.get_param("debug/viz_rollouts/print_stats", False)
        )

    def viz_rollouts(
        self, cost, cost2go, collision_cost, obs_dist_cost, smoothness, poses
    ):
        non_colliding = (collision_cost == 0).nonzero()

        if non_colliding.size()[0] > 0:

            def print_n(c, poses, ns, cmap="coolwarm"):
                _, all_idx = torch.sort(c)

                n = min(self.n_viz, len(c))
                idx = all_idx[:n] if n > -1 else all_idx
                self.viz_paths_cmap(poses[idx], c[idx], ns=ns, cmap=cmap)

            p_non_colliding = poses[non_colliding].squeeze()
            print_n(cost[non_colliding].squeeze(), p_non_colliding, ns="final_result")
            print_n(cost2go[non_colliding].squeeze(), p_non_colliding, ns="cost2go")
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
                smoothness[non_colliding].squeeze(), p_non_colliding, ns="smoothness"
            )

            if self.print_stats:
                _, all_sorted_idx = torch.sort(cost[non_colliding].squeeze())
                n = min(self.n_viz, len(all_sorted_idx))
                idx = all_sorted_idx[:n] if n > -1 else all_sorted_idx

                print("Final Result")
                print(cost[idx])
                print("Cost 2 Go")
                print(cost2go[idx])
                print("Collision Cost")
                print(collision_cost[idx])
                print("Obstacle Distance Cost")
                print(obs_dist_cost[idx])
                print("Smoothness")
                print(smoothness[idx])

    def viz_paths_cmap(self, poses, costs, ns="paths", cmap="plasma", scale=0.03):
        max_c = torch.max(costs)
        min_c = torch.min(costs)

        norm = colors.Normalize(vmin=min_c, vmax=max_c)

        cmap = cm.get_cmap(name=cmap)

        def colorfn(cost):
            r, g, b, a = 0.0, 0.0, 0.0, 1.0
            if cost == min_c:
                return r, g, b, a
            col = cmap(norm(cost))
            r, g, b = col[0], col[1], col[2]
            if len(col) > 3:
                a = col[3]
            return r, g, b, a

        return self.viz_paths(poses, costs, colorfn, ns, scale)

    def viz_paths(self, poses, costs, colorfn, ns="paths", scale=0.03):
        """
            poses should be an array of trajectories to plot in rviz
            costs should have the same dimensionality as poses.size()[0]
            colorfn maps a point to an rgb tuple of colors
        """
        assert poses.size()[0] == costs.size()[0]

        markers = MarkerArray()

        for i, (traj, cost) in enumerate(zip(poses, costs)):
            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = rospy.Time.now()
            m.ns = ns
            m.id = i
            m.type = m.LINE_STRIP
            m.action = m.ADD
            m.pose.position.x = 0
            m.pose.position.y = 0
            m.pose.position.z = 0
            m.pose.orientation.x = 0.0
            m.pose.orientation.y = 0.0
            m.pose.orientation.z = 0.0
            m.pose.orientation.w = 1.0
            m.scale.x = scale
            m.color.r, m.color.g, m.color.b, m.color.a = colorfn(cost)

            for t in traj:
                p = Point()
                p.x, p.y = t[0], t[1]
                m.points.append(p)

            markers.markers.append(m)

        for i in range(len(poses), rospy.get_param("~K"), 1):
            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = rospy.Time.now()
            m.ns = ns
            m.id = i
            m.type = m.LINE_STRIP
            m.action = m.DELETE
            markers.markers.append(m)

        self.traj_pub.publish(markers)
