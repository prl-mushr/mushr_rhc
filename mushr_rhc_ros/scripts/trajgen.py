import numpy as np
from scipy import signal


def saw():
    t = np.linspace(-4, 4, 50)
    saw = signal.sawtooth(0.5 * np.pi * t)
    configs = [[x, y, 0] for (x, y) in zip(t, saw)]
    return configs


def wave():
    ts = np.linspace(-5, 5, 50)
    period = 0.8
    ys = np.sin(period * ts)
    theta = np.cos(period * ts)
    configs = [[x, y, _theta] for (x, y, _theta) in zip(ts, ys, theta)]
    return configs


def circle():
    waypoint_sep = 0.1
    radius = 2.5
    center = [0, radius]
    num_points = int((2 * radius * np.pi) / waypoint_sep)
    thetas = np.linspace(-1 * np.pi / 2, 2 * np.pi - (np.pi / 2), num_points)
    poses = [[radius * np.cos(theta) + center[0], radius * np.sin(theta) + center[1], theta + (np.pi / 2)] for theta in thetas]
    return poses


def left_turn(turn_rad, pathlen=4.0):
    waypoint_sep = 0.1
    turn_radius = turn_rad
    straight_len = 1.0
    turn_center = [straight_len, turn_radius]
    straight_xs = np.linspace(0, straight_len, int(straight_len / waypoint_sep))
    straight_poses = [[x, 0, 0] for x in straight_xs]

    num_turn_points = int((turn_radius * np.pi * 0.5) / waypoint_sep)
    thetas = np.linspace(-1 * np.pi / 2, 0, num_turn_points)

    turn_poses = [[turn_radius * np.cos(theta) + turn_center[0], turn_radius * np.sin(theta) + turn_center[1], theta + (np.pi / 2)] for theta in thetas]
    poses = straight_poses + turn_poses
    return poses[:min(int(pathlen / waypoint_sep), len(poses))]


def left_kink(turn_rad, pathlen=4.0):
    waypoint_sep = 0.1
    straight_len = 1.0
    turn1_center = [straight_len, turn_rad]

    straight_xs = np.linspace(0, straight_len, int(straight_len / waypoint_sep))
    straight_poses = [[x, 0, 0] for x in straight_xs]

    num_turn1_points = int((turn_rad * np.pi / 4) / waypoint_sep)
    turn1_thetas = np.linspace(-1 * np.pi / 2, -np.pi / 4, num_turn1_points)
    turn1_poses = [[turn_rad * np.cos(theta) + turn1_center[0], turn_rad * np.sin(theta) + turn1_center[1], theta + (np.pi / 2)] for theta in turn1_thetas]

    kink_xs = np.linspace(0, turn_rad * 2, int(turn_rad * 2 / waypoint_sep))
    kink = [[x * np.cos(np.pi / 4) + turn1_poses[-1][0], x * np.sin(np.pi / 4) + turn1_poses[-1][1], np.pi / 4] for i, x in enumerate(kink_xs)]

    poses = straight_poses + turn1_poses + kink
    return poses[:min(int(pathlen / waypoint_sep), len(poses))]


def right_turn(turn_rad, pathlen=4.0):
    waypoint_sep = 0.1
    turn_radius = turn_rad
    straight_len = 1.0
    turn_center = [straight_len, -turn_radius]
    straight_xs = np.linspace(0, straight_len, int(straight_len / waypoint_sep))
    straight_poses = [[x, 0, 0] for x in straight_xs]
    num_turn_points = int((turn_radius * np.pi * 0.5) / waypoint_sep)
    thetas = np.linspace(1 * np.pi / 2, 0, num_turn_points)
    turn_poses = [[turn_radius * np.cos(theta) + turn_center[0], turn_radius * np.sin(theta) + turn_center[1], theta - (np.pi / 2)] for theta in thetas]
    poses = straight_poses + turn_poses
    return poses[:min(int(pathlen / waypoint_sep), len(poses))]


def right_kink(turn_rad, pathlen=4.0):
    waypoint_sep = 0.1
    straight_len = 1.0
    turn1_center = [straight_len, -turn_rad]

    straight_xs = np.linspace(0, straight_len, int(straight_len / waypoint_sep))
    straight_poses = [[x, 0, 0] for x in straight_xs]

    num_turn1_points = int((turn_rad * np.pi / 4) / waypoint_sep)
    turn1_thetas = np.linspace(np.pi / 2, np.pi / 4, num_turn1_points)
    turn1_poses = [[turn_rad * np.cos(theta) + turn1_center[0], turn_rad * np.sin(theta) + turn1_center[1], theta - (np.pi / 2)] for theta in turn1_thetas]

    kink_xs = np.linspace(0, turn_rad * 2, int(turn_rad * 2 / waypoint_sep))
    kink = [[x * np.cos(-np.pi / 4) + turn1_poses[-1][0], x * np.sin(-np.pi / 4) + turn1_poses[-1][1], -np.pi / 4] for i, x in enumerate(kink_xs)]

    poses = straight_poses + turn1_poses + kink
    return poses[:min(int(pathlen / waypoint_sep), len(poses))]


def straight_line(pathlen=2.0):
    waypoint_sep = 0.1
    straight_len = pathlen
    straight_xs = np.linspace(0, straight_len, int(straight_len / waypoint_sep))
    straight_poses = [[x, 0, 0] for x in straight_xs]
    return straight_poses
