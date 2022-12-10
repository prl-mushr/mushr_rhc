import numpy as np
import rospy

from controller import BaseController


class PIDController(BaseController):
    def __init__(self):
        super(PIDController, self).__init__()

        self.reset_params()
        self.reset_state()

    def get_reference_index(self, pose):
        '''
        get_reference_index finds the index i in the controller's path
            to compute the next control control against
        input:
            pose - current pose of the car, represented as [x, y, heading]
        output:
            i - referencence index
        '''
        with self.path_lock:
            pose = np.array(pose)
            diff = self.path[:, :3] - pose
            dist = np.linalg.norm(diff[:, :2], axis=1)
            index = dist.argmin()
            index += int(self.waypoint_lookahead / self.waypoint_diff)
            index = min(index, len(self.path)-1)
            return index

    def get_control(self, pose, index):
        '''
        get_control - computes the control action given an index into the
            reference trajectory, and the current pose of the car.
            Note: the output velocity is given in the reference point.
        input:
            pose - the vehicle's current pose [x, y, heading]
            index - an integer corresponding to the reference index into the
                reference path to control against
        output:
            control - [velocity, steering angle]
        '''
        error = self.get_error(pose, index)
        self.gain_p = error[1]
        self.gain_d = np.sin(np.arctan(error[1]/error[0]))

        control = self.kp * self.gain_p + self.kd * self.gain_d

        return [self.path[index, 3], float(control)]

    def reset_state(self):
        '''
        Utility function for resetting internal states.
        '''
        with self.path_lock:
            pass

    def reset_params(self):
        '''
        Utility function for updating parameters which depend on the ros parameter
            server. Setting parameters, such as gains, can be useful for interative
            testing.
        '''
        with self.path_lock:
            self.kp = float(rospy.get_param("/pid/kp", 0.15))
            self.kd = float(rospy.get_param("/pid/kd", 0.2))
            self.finish_threshold = float(rospy.get_param("/pid/finish_threshold", 0.2))
            self.exceed_threshold = float(rospy.get_param("/pid/exceed_threshold", 4.0))
            # Average distance from the current reference pose to lookahed.
            self.waypoint_lookahead = float(rospy.get_param("/pid/waypoint_lookahead", 0.6))
