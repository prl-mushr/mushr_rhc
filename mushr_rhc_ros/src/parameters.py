import rospy


class RosParams:
    def get_str(self, path, default=None, global_=False):
        return str(self._get_raw(global_, path, default))

    def get_dict(self, path, default=None, global_=False):
        return self._get_raw(global_, path, default)

    def get_int(self, path, default=None, global_=False):
        return int(self._get_raw(global_, path, default))

    def get_float(self, path, default=None, global_=False):
        return float(self._get_raw(global_, path, default))

    def get_bool(self, path, default=None, global_=False):
        return bool(self._get_raw(global_, path, default))

    def _get_raw(self, global_, path, default=None):
        if global_:
            prefix = "/"
        else:
            prefix = "~"

        rospath = prefix + path
        if default is None:
            return rospy.get_param(rospath)
        else:
            return rospy.get_param(rospath, default)


class DictParams:
    def __init__(self, params):
        self.params = params

    def get_str(self, path, default=None, global_=False):
        return str(self._get_raw(global_, path, default))

    def get_dict(self, path, default=None, global_=False):
        return self._get_raw(global_, path, default)

    def get_int(self, path, default=None, global_=False):
        return int(self._get_raw(global_, path, default))

    def get_float(self, path, default=None, global_=False):
        return float(self._get_raw(global_, path, default))

    def get_bool(self, path, default=None, global_=False):
        return bool(self._get_raw(global_, path, default))

    def _get_raw(self, global_, path, default=None):
        prefix = ""
        if global_:
            prefix = "/"

        rospath = prefix + path
        if path in self.params:
            return self.params[rospath]
        else:
            return default
