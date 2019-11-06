# Copyright (c) 2019, The Personal Robotics Lab, The MuSHR Team, The Contributors of MuSHR
# License: BSD 3-Clause. See LICENSE.md file in root directory.

import os

import numpy as np
import rospy
from geometry_msgs.msg import Point
from nav_msgs.msg import MapMetaData
from nav_msgs.srv import GetMap
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker

import mushr_rhc
import mushr_rhc.cost
import mushr_rhc.model
import mushr_rhc.trajgen
import mushr_rhc.value
import mushr_rhc.worldrep
import rosvizpath
import utils

motion_models = {"kinematic": mushr_rhc.model.Kinematics, "mujoco": mushr_rhc.model.MujocoSim}

trajgens = {
    "tl": mushr_rhc.trajgen.TL,
    "dispersion": mushr_rhc.trajgen.Dispersion,
    "mppi": mushr_rhc.trajgen.MXPI
}

cost_functions = {"waypoints": mushr_rhc.cost.Waypoints, "block_push": mushr_rhc.cost.BlockPush}

value_functions = {"simpleknn": mushr_rhc.value.SimpleKNN}

world_reps = {"simple": mushr_rhc.worldrep.Simple}


def viz_halton(hp, dsts):
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
    max_d = np.max(dsts)
    for i, pts in enumerate(hp):
        p = Point()
        c = ColorRGBA()
        c.a = 1
        c.g = int(255.0 * dsts[i] / max_d)
        p.x, p.y = pts[0], pts[1]
        m.points.append(p)
        m.colors.append(c)

    pub = rospy.Publisher("~markers", Marker, queue_size=1)
    pub.publish(m)


class RHCBase(object):
    def __init__(self, dtype, params, logger):
        self.dtype = dtype
        self.params = params
        self.logger = logger

        rospy.Subscriber(
            "/map_metadata", MapMetaData, self.cb_map_metadata, queue_size=1
        )
        self.map_data = None

    def load_controller(self):
        m = self.get_model()
        tg = self.get_trajgen()
        cf = self.get_cost_fn()

        return mushr_rhc.MPC(self.params, self.logger, self.dtype, m, tg, cf)

    def get_model(self):
        mname = self.params.get_str("model_name", default="kinematic")
        if mname not in motion_models:
            self.logger.fatal("model '{}' is not valid".format(mname))

        return motion_models[mname](self.params, self.logger, self.dtype)

    def get_trajgen(self):
        tgname = self.params.get_str("trajgen_name", default="tl")
        if tgname not in trajgens:
            self.logger.fatal("trajgen '{}' is not valid".format(tgname))

        return trajgens[tgname](self.params, self.logger, self.dtype)

    def get_cost_fn(self):
        cfname = self.params.get_str("cost_fn_name", default="waypoints")
        if cfname not in cost_functions:
            self.logger.fatal("cost_fn '{}' is not valid".format(cfname))

        wrname = self.params.get_str("world_rep_name", default="simple")
        if wrname not in world_reps:
            self.logger.fatal("world_rep '{}' is not valid".format(wrname))

        self.logger.debug("Waiting for map metadata")
        while self.map_data is None:
            rospy.sleep(0.1)
        self.logger.debug("Recieved map metadata")

        wr = world_reps[wrname](self.params, self.logger, self.dtype, self.map_data)

        vfname = self.params.get_str("value_fn_name", default="simpleknn")
        if vfname not in value_functions:
            self.logger.fatal("value_fn '{}' is not valid".format(vfname))

        vf = value_functions[vfname](
            self.params, self.logger, self.dtype, self.map_data, viz_halton
        )

        viz_rollouts_fn = None
        print rospy.get_param("~debug/flag/viz_rollouts", False)
        if bool(rospy.get_param("~debug/flag/viz_rollouts", False)):
            print "VIS ROLLOUTS"
            viz_rollouts_fn = rosvizpath.VizPaths().viz_rollouts

        return cost_functions[cfname](
            self.params, self.logger, self.dtype, self.map_data, wr, vf, viz_rollouts_fn
        )

    def cb_map_metadata(self, msg):
        default_map_name = "default"
        map_file = self.params.get_str(
            "map_file", default=default_map_name, global_=True
        )
        name = os.path.splitext(os.path.basename(map_file))[0]

        if name is default_map_name:
            rospy.logwarn(
                "Default map name being used, will be corrupted on map change. "
                + "To fix, set '/map_file' parameter with map_file location"
            )

        x, y, angle = utils.rospose_to_posetup(msg.origin)
        self.map_data = mushr_rhc.MapInfo(
            name=name,
            resolution=msg.resolution,
            origin_x=x,
            origin_y=y,
            orientation_angle=angle,
            width=msg.width,
            height=msg.height,
            get_map_data=self.get_map,
        )

    def get_map(self):
        srv_name = self.params.get_str("static_map", default="/static_map")
        self.logger.debug("Waiting for map service")
        rospy.wait_for_service(srv_name)
        self.logger.debug("Map service started")

        map_msg = rospy.ServiceProxy(srv_name, GetMap)().map
        return map_msg.data
