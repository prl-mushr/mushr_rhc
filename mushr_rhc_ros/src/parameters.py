import rospy

class RosParams:
    def get_str(self, path, default=None, global_=False):
        return str(_get_raw(global_, path, default))

    def get_dict(self, path, default=None, global_=False):
        return _get_raw(global_, path, default)

    def get_float(self, path, default=None, global_=False):
        return float(_get_raw(global_, path, default))

    def get_bool(self, path, default=None, global_=False):
        return bool(_get_raw(global_, path, default))

    def _get_raw(self, global_, path, default=None):
        if global_:
            prefix = "/"
        else:
            prefix = "~"

        rospath = prefix + path
        try:
            return rospy.get_param(rospath, default)
        except KeyError:
            rospy.log_err("param '{}' not set (and no default provided)".format(rospath))
            sys.exit(1)
