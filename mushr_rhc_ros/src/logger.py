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


class StdLog:
    def debug(self, msg, *args):
        print "[DEBUG] " + msg.format(*args)

    def warn(self, msg, *args):
        print "[WARN] " + msg.format(*args)

    def info(self, msg, *args):
        print "[INFO] " + msg.format(*args)

    def err(self, msg, *args):
        print "[ERROR] " + msg.format(*args)

    def fatal(self, msg, *args):
        print "[FATAL] " + msg.format(*args)
        exit(1)
