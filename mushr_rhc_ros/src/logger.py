import rospy


class RosLog:
    def debug(self, msg, *args):
        rospy.logdebug(msg, *args)

    def warn(self, msg, *args):
        rospy.logwarn(msg, *args)

    def info(self, msg, *args):
        rospy.loginfo(msg, *args)

    def err(self, msg, *args):
        rospy.logerr(msg, *args)

    def fatal(self, msg, *args):
        rospy.logfatal(msg, *args)
