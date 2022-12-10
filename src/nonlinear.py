import numpy as np
import rospy

from controller import BaseController


# Uses Proportional-Differential Control from
# https://www.a1k0n.net/2018/11/13/fast-line-following.html
class NonLinearController(BaseController):
    def __init__(self):
        super(NonLinearController, self).__init__()
        self.reset_params()
        self.reset_state()

    def get_reference_index(self, pose):
        print(self.path)
        with self.path_lock:
            # TODO: compute index of next point to optimize against
            pose = np.array(pose)
            diff = self.path[:, :3] - pose
            dist = np.linalg.norm(diff[:, :2], axis=1)
            index = dist.argmin()
            index += int(self.lookahead / self.waypoint_diff)
            index = min(index, len(self.path)-1)
            print("INDEX " + str(index))
            return index

    def get_control(self, pose, index):
        if self.prev_error is None:
            print("No prev error")
            self.prev_error = 0.0
        if self.prev_ref_heading is None:
            print("No prev error")
            self.prev_ref_heading = self.path[index][2]
            return [0.0, 0.0]

        error = self.get_error(pose, index)
        angle = np.arctan(error[1]/error[0])
        c_e, s_e = np.cos(angle), np.sin(angle)
        curvature = (self.path[index][2] - self.prev_ref_heading) * 100

        control = (c_e / 1 - curvature * error[1]) \
            * (-self.kp * ((error[1] * (c_e)**2)/(1 - curvature * error[1]))
                + (s_e * (curvature * s_e - self.kd * c_e)))

        if np.absolute(curvature) < self.k_min:
            v = self.path[index, 3]
        else:
            v = min(np.absolute(self.a_l/control) ** 0.5, self.speed)

        self.prev_error = self.gain_p
        self.prev_ref_heading = self.path[index][2]
        print("km        " + str(self.k_min))
        print("k         " + str(curvature))
        print("gp (error)" + str(self.gain_p))
        print("kp        " + str(self.kp))
        print("gd        " + str(self.gain_d))
        print("kd        " + str(self.kd))
        print("CONTROL   " + str(control))
        print("---")
        return [v, float(control)]

    def reset_state(self):
        with self.path_lock:
            self.prev_error = None
            self.prev_ref_heading = None
            self.gain_p = 0.0
            self.gain_d = 0.0

    def reset_params(self):
        with self.path_lock:
            self.kp = float(rospy.get_param("/nl/kp", -0.1))
            self.kd = float(rospy.get_param("/nl/kd", -1.0))
            self.speed = float(rospy.get_param("/nl/speed", 1.0))
            self.finish_threshold = float(rospy.get_param("/nl/finish_threshold", 1.0))
            self.exceed_threshold = float(rospy.get_param("/nl/exceed_threshold", 4.0))
            self.lookahead = float(rospy.get_param("/nl/lookahead", 1.5))
            self.a_l = float(rospy.get_param("/nl/a_l", 1.5))
            self.k_min = self.a_l / (self.speed ** 2)
