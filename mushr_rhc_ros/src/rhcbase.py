# Copyright (c) 2019, The Personal Robotics Lab, The MuSHR Team, The Contributors of MuSHR
# License: BSD 3-Clause. See LICENSE.md file in root directory.

import os

import rospy
from nav_msgs.msg import MapMetaData
from nav_msgs.srv import GetMap

import librhc
import librhc.cost as cost
import librhc.model as model
import librhc.trajgen as trajgen
import librhc.types as types
import librhc.worldrep as worldrep
import utils

motion_models = {"kinematic": model.Kinematics}

trajgens = {"tl": trajgen.TL, "dispersion": trajgen.Dispersion}

cost_functions = {"waypoints": cost.Waypoints}

world_reps = {"simple": worldrep.Simple}


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
        tg = self.get_trajgen(m)
        cf = self.get_cost_fn()

        return librhc.MPC(self.params, self.logger, self.dtype, m, tg, cf)

    def get_model(self):
        mname = self.params.get_str("model_name", default="kinematic")
        if mname not in motion_models:
            self.logger.fatal("model '{}' is not valid".format(mname))

        return motion_models[mname](self.params, self.logger, self.dtype)

    def get_trajgen(self, model):
        tgname = self.params.get_str("trajgen_name", default="tl")
        if tgname not in trajgens:
            self.logger.fatal("trajgen '{}' is not valid".format(tgname))

        return trajgens[tgname](self.params, self.logger, self.dtype, model)

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

        return cost_functions[cfname](
            self.params, self.logger, self.dtype, self.map_data, wr
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
        self.map_data = types.MapData(
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
