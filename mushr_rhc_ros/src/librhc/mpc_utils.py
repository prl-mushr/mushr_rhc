#!/usr/bin/env python

import rospy
import numpy as np

import torch
import csv

from std_msgs.msg import Header, Float64
from m3pi.msg import Particle, ParticleArray, TLCosts
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Pose, PoseStamped, PoseArray, Quaternion, PolygonStamped, Polygon, Point32, PoseWithCovarianceStamped, PointStamped
import tf.transformations
import tf
import matplotlib.pyplot as plt


def angle_to_quaternion(angle):
    """Convert an angle in radians into a quaternion _message_."""
    return Quaternion(*tf.transformations.quaternion_from_euler(0, 0, angle))


def quaternion_to_angle(q):
    """Convert a quaternion _message_ into an angle in radians.
    The angle represents the yaw.
    This is not just the z component of the quaternion."""
    x, y, z, w = q.x, q.y, q.z, q.w
    roll, pitch, yaw = tf.transformations.euler_from_quaternion((x, y, z, w))
    return yaw


# # Scary experimental canadian strategy:
# def _new_imod(self, a):
#     return torch.fmod(self, a, out=self)
# torch.Tensor.__imod__ = _new_imod

# modifies angles to be in range [-pi, pi]
def clamp_angle(angles):
    angles += np.pi
    angles %= (2 * np.pi)
    angles -= np.pi
    return angles


def clamp_angle_tensor_(angles):
    angles += np.pi
    angles.remainder_(2 * np.pi)
    angles -= np.pi


def rotation_matrix(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.matrix([[c, -s], [s, c]])

# CSV:
#   x,y
#   1,1
#   2,5
#   ...
# Return a list of configs [[x, y, 0], [x, y, 0]] in world frame


def load_csv_to_configs(file_name, csv_mode="world", map_info=None):
    configs = []
    with open(file_name, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # ignore title row
            if row[0] == 'x':
                continue

            config = [float(row[0]), float(row[1]), 0.0]
            # pixel to world
            if csv_mode == "pixel":
                config[1] = map_info.height - config[1]  # FLIP
                pose = our_map_to_world(config, map_info)
                config[0] = pose[0]
                config[1] = pose[1]
            configs.append(config)

    return configs


def posewithcovariancestamped_to_config(pwcs):
    x = pwcs.pose.pose.position.x
    y = pwcs.pose.pose.position.y
    th = quaternion_to_angle(pwcs.pose.pose.orientation)
    return [x, y, th]


def pose_to_config(pose):
    x = pose.position.x
    y = pose.position.y
    th = quaternion_to_angle(pose.orientation)
    return [x, y, th]


def pose_to_config_torch(pose):
    return torch.Tensor(pose_to_config(pose))


def config_pose_to_config(pose):
    x = pose.position.x
    y = pose.position.y
    th = pose.position.z
    return [x, y, th]


def parse_particle_array_msg(msg, dtype):
    particles = []
    weights = []
    for particle in msg.particles:
        config = [particle.x, particle.y, particle.theta]
        particles.append(config)
        weights.append(particle.weight)
    return torch.Tensor(particles).type(
        dtype), torch.Tensor(weights).type(dtype)


def create_tl_costs_msg(init_pose, costs):
    costs_np = map(Float64, np.nan_to_num(costs).astype(np.float64))
    msg = TLCosts()
    msg.header = make_header('map')
    msg.costs = costs_np
    msg.x = Float64(init_pose[0, 0])
    msg.y = Float64(init_pose[0, 1])
    msg.theta = Float64(init_pose[0, 2])
    return msg


def poses_into_buffer(buf, poses):
    map(lambda x: buf[x[0]].copy_(
        pose_to_config_torch(x[1])), enumerate(poses))


def posestamped_to_config(posestamped):
    x = posestamped.pose.position.x
    y = posestamped.pose.position.y
    th = quaternion_to_angle(posestamped.pose.orientation)
    return [x, y, th]


def config_to_pose(config):
    pose = Pose()
    pose.position.x = config[0]
    pose.position.y = config[1]
    pose.orientation = angle_to_quaternion(config[2])
    return pose


def config_to_posestamped(config):
    pose = PoseStamped()
    pose.header = make_header('map')
    pose.pose.position.x = config[0]
    pose.pose.position.y = config[1]
    pose.pose.orientation = angle_to_quaternion(config[2])
    return pose


def particle_to_posestamped(particle, frame_id):

    pose = PoseStamped()
    pose.header = make_header(frame_id)
    pose.pose.position.x = particle[0]
    pose.pose.position.y = particle[1]
    pose.pose.orientation = angle_to_quaternion(particle[2])
    return pose


def particle_to_posewithcovariancestamped(particle, frame_id):

    pose = PoseWithCovarianceStamped()
    pose.header = make_header(frame_id)
    pose.pose.pose.position.x = particle[0]
    pose.pose.pose.position.y = particle[1]
    pose.pose.pose.orientation = angle_to_quaternion(particle[2])
    return pose


def particle_to_pose(particle):
    pose = Pose()
    pose.position.x = particle[0]
    pose.position.y = particle[1]
    pose.orientation = angle_to_quaternion(particle[2])
    return pose


def particles_to_poses(particles):
    return map(particle_to_pose, particles)


def make_header(frame_id, stamp=None):
    if stamp is None:
        stamp = rospy.Time.now()
    header = Header()
    header.stamp = stamp
    header.frame_id = frame_id
    return header


def point(npt):
    pt = Point32()
    pt.x = npt[0]
    pt.y = npt[1]
    return pt


def points(arr):
    return map(point, arr)


def map_to_world(poses, map_info):
    scale = map_info.resolution
    angle = quaternion_to_angle(map_info.origin.orientation)

    # rotate

    # rotation
    c, s = np.cos(angle), np.sin(angle)
    # we need to store the x coordinates since they will be overwritten
    temp = np.copy(poses[:, 0])
    poses[:, 0] = c * poses[:, 0] - s * poses[:, 1]
    poses[:, 1] = s * temp + c * poses[:, 1]

    # scale
    poses[:, :2] *= float(scale)

    # translate
    poses[:, 0] += map_info.origin.position.x
    poses[:, 1] += map_info.origin.position.y
    poses[:, 2] += angle


def world_to_map(poses, map_info):
    # equivalent to map_to_grid(world_to_map(poses))
    # operates in place
    scale = map_info.resolution
    angle = -quaternion_to_angle(map_info.origin.orientation)

    # translation
    poses[:, 0] -= map_info.origin.position.x
    poses[:, 1] -= map_info.origin.position.y

    # scale
    poses[:, :2] *= (1.0 / float(scale))

    # rotation
    c, s = np.cos(angle), np.sin(angle)
    # we need to store the x coordinates since they will be overwritten
    temp = np.copy(poses[:, 0])
    poses[:, 0] = c * poses[:, 0] - s * poses[:, 1]
    poses[:, 1] = s * temp + c * poses[:, 1]
    poses[:, 2] += angle


def world_to_map_torch(poses, map_info, angle, c, s):
    # equivalent to map_to_grid(world_to_map(poses))
    # operates in place
    scale = map_info.resolution

    # translation
    xs = poses[:, 0]
    ys = poses[:, 1]
    xs -= map_info.origin.position.x
    ys -= map_info.origin.position.y

    # scale
    poses[:, :2] *= (1.0 / float(scale))

    # we need to store the x coordinates since they will be overwritten
    temp = xs.clone()
    xs *= c
    # xs -= s * ys
    xs -= ys * s
    ys *= c
    ys += temp * s
    poses[:, 2] += angle
