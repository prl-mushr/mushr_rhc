import torch
import math
import threading
import Tkinter as tk
import mushr_rhc.utils as utils


class BlockPush:
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

        self.reset()

    def reset(self):
        self.T = self.params.get_int("T", default=15)
        self.K = self.params.get_int("K", default=62)
        self.NPOS = self.params.get_int("npos", default=3)

        time_horizon = utils.get_time_horizon(self.params)
        self.dt = time_horizon / self.T

        self.goal_lock = threading.RLock()
        with self.goal_lock:
            self.goal = None
        self.goal_threshold = self.params.get_float("xy_threshold", default=0.2)
        self.dist_horizon = utils.get_distance_horizon(self.params)

        self.dist_w = self.params.get_float("cost_fn/dist_w", default=1.0)
        self.car_obs_dist_w = self.params.get_float(
            "cost_fn/car_obs_dist_w", default=5.0
        )
        self.block_obs_dist_w = self.params.get_float(
            "cost_fn/car_obs_dist_w", default=5.0
        )
        self.cost2go_w = self.params.get_float("cost_fn/cost2go_w", default=1.0)
        self.bounds_cost = self.params.get_float("cost_fn/bounds_cost", default=100.0)

        self.world_rep.reset()

        self.a_diff_w = 1.0
        self.block_car_dist_w = 1.0
        self.block_car_dist_shift = 2.0
        self.debug_vis = False
        self.debug_with_sliders = False
        if self.debug_vis:
            if self.debug_with_sliders:
                threading.Thread(target=self.display_window).start()

    def display_window(self):
        master = tk.Tk()
        t = tk.Text(master, height=1)
        t.pack()
        t.insert(tk.END, "a_diff_w")
        t.config(state="disabled")
        self.a_diff_scale = tk.Scale(
            master, from_=0.0, to=10.0, length=300, orient=tk.HORIZONTAL, resolution=0.1
        )
        self.a_diff_scale.set(self.a_diff_w)
        self.a_diff_scale.pack()

        t = tk.Text(master, height=1)
        t.pack()
        t.insert(tk.END, "block_car_dist_w")
        t.config(state="disabled")
        self.block_car_dist_scale = tk.Scale(
            master, from_=0.0, to=10.0, length=300, orient=tk.HORIZONTAL, resolution=0.1
        )
        self.block_car_dist_scale.set(self.block_car_dist_w)
        self.block_car_dist_scale.pack()

        t = tk.Text(master, height=1)
        t.pack()
        t.insert(tk.END, "block_car_dist_shift")
        t.config(state="disabled")
        self.block_car_dist_shift_scale = tk.Scale(
            master, from_=0.0, to=10.0, length=300, orient=tk.HORIZONTAL, resolution=0.1
        )
        self.block_car_dist_shift_scale.set(self.block_car_dist_shift)
        self.block_car_dist_shift_scale.pack()

        t = tk.Text(master, height=1)
        t.pack()
        t.insert(tk.END, "cost2go_w")
        t.config(state="disabled")
        self.cost2go_w_scale = tk.Scale(
            master, from_=0.0, to=10.0, length=300, orient=tk.HORIZONTAL, resolution=0.1
        )
        self.cost2go_w_scale.set(self.cost2go_w)
        self.cost2go_w_scale.pack()

        self.weights_text = tk.Text(master, height=self.K + 2, width=150)
        self.weights_text.pack()

        tk.mainloop()

    def get_weights(self):
        if self.debug_with_sliders:
            self.a_diff_w = self.a_diff_scale.get()
            self.block_car_dist_w = self.block_car_dist_scale.get()
            self.block_car_dist_shift = self.block_car_dist_shift_scale.get()
            self.cost2go_w = self.cost2go_w_scale.get()

    def apply(self, poses):
        assert poses.size() == (self.K, self.T, self.NPOS)
        # Currently the goal is just a place for the block
        # assert goal.size() == (self.NPOS,)
        with self.goal_lock:
            goal = self.goal
        assert goal.size() == (3,)

        if self.debug_vis:
            self.get_weights()

        s_block_goal_vec = goal[:2] - poses[0, 0, 3:5]  # (2, )
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
        f_block_goal_vec = goal[:2] - poses[:, final_idx, 3:5]  # (K, 2)
        f_block_car_vec = poses[:, final_idx, :2] - poses[:, final_idx, 3:5]  # (K, 2)

        f_block_car_dist = f_block_car_vec.pow(2).sum(dim=1).pow_(0.5)  # (K,)
        f_block_goal_dist = f_block_goal_vec.pow(2).sum(dim=1).pow_(0.5)  # (K,)

        angles = f_block_goal_vec.mul(f_block_car_vec).sum(dim=1)
        angles.div_(f_block_goal_dist).div_(f_block_car_dist).acos_()
        angles[angles < 0] += 2 * math.pi

        if not (
            3.0 / 4.0 * math.pi <= car_goal_angle
            and car_goal_angle <= 5.0 / 4.0 * math.pi
        ):
            a_diff = (angles - math.pi).abs_()
            # want trajectory that points opposite to block -> goal
            # AND at least 2m (or something like this) away from block
            all_poses = poses.view(self.K * self.T, self.NPOS)
            dist_cost = (
                (all_poses[:, :2] - all_poses[:, 3:5]).pow(2).sum(dim=1).pow_(0.5)
            )
            dist_cost.sub_(self.block_car_dist_shift).pow_(2)
            # dist_cost = dist_cost.view(self.K, self.T).sum(dim=1)
            dist_cost = dist_cost.view(self.K, self.T)[:, self.T - 1]

            a_diff.mul_(self.a_diff_w)
            dist_cost.mul_(self.block_car_dist_w)
            result = a_diff.add(dist_cost)

            if self.viz_rollouts_fn:
                self.viz_rollouts_fn(
                    result,
                    poses,
                    angles=angles,
                    car_block_angle_diff=a_diff,
                    block_car_dist_cost=dist_cost,
                    block_car_dist=f_block_car_dist,
                )

                if self.debug_vis:
                    if self.debug_with_sliders:
                        text = "NAV2BLOCK PHASE\n"
                        text += "result\t\ta_diff\t\tdist_cost\n"
                        for v in zip(result, a_diff, dist_cost):
                            text += "%f\t\t%f\t\t%f\n" % (v[0], v[1], v[2])

                        self.weights_text.insert("1.0", text)

        elif not (
            s_block_car_dist < 0.5
            # and (
            #     car_goal_angle <= 1.0 / 7.0 * math.pi
            #     or car_goal_angle >= 13.0 / 7.0 * math.pi
            # )
        ):
            # want trajectory that points in the same direction as the block to the goal
            # AND as close to the block as possible (requires first objective tho)

            dist_cost = (
                (poses[:, final_idx, :2] - poses[:, final_idx, 3:5]).pow(2).sum(dim=1)
            )  # .pow_(0.5)
            cost2go = self.value_fn.get_value(poses[:, final_idx, 3:]).mul(
                self.cost2go_w
            )
            result = cost2go.add(dist_cost)

            if self.viz_rollouts_fn:
                self.viz_rollouts_fn(
                    result,
                    poses,
                    angles=angles,
                    block_car_dist_cost=dist_cost,
                    block_car_dist=f_block_car_dist,
                )

                if self.debug_vis:
                    if self.debug_with_sliders:
                        text = "COST2GO PHASE\n"
                        text += "result\t\tcost2go\t\tdist_cost\n"
                        for v in zip(result, cost2go, dist_cost):
                            text += "%f\t\t%f\t\t%f\n" % (v[0], v[1], v[2])

                        self.weights_text.insert("1.0", text)

        else:
            cost2go = self.value_fn.get_value(poses[:, final_idx, 3:]).mul(
                self.cost2go_w
            )
            result = cost2go.add(f_block_goal_dist)

            if self.viz_rollouts_fn:
                self.viz_rollouts_fn(
                    result,
                    poses,
                )

                if self.debug_vis:
                    if self.debug_with_sliders:
                        text = "GET TO THE GOAL\n"

                        self.weights_text.insert("1.0", text)

            # raw_input("Press enter:")
        return result, False  # the false is backward trajectories

    def set_goal(self, goal):
        """
        Args:
        goal [(3,) tensor] -- Goal in "world" coordinates
        """
        assert goal.size() == (3,)

        with self.goal_lock:
            self.goal = goal
            return self.value_fn.set_goal(goal)

    def dist_to_goal(self, state):
        # use block, not car as dist to goal
        with self.goal_lock:
            if self.goal is None:
                return False
            return self.goal[:2].dist(state[3:5])

    def at_goal(self, state):
        """
        Args:
        state [(3,) tensor] -- Current position in "world" coordinates
        """
        with self.goal_lock:
            if self.goal is None:
                return False
            return self.dist_to_goal(state) < self.goal_threshold

    def get_desired_speed(self, desired_speed, state):
        return desired_speed
