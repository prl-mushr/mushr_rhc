import numpy as np
import rospy
from controller import BaseController


class PurePursuitController(BaseController):
    def __init__(self):
        super(PurePursuitController, self).__init__()

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
            for i in range(index, len(self.path)):
                if dist[i] > self.pose_lookahead:
                    return i
            return len(self.path)-1

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
        cte = error[1]
        control = (2 * cte) / (self.pose_lookahead ** 2)
        return [self.path[index, 3], float(control)]

    def reset_state(self):
        '''
        Utility function for resetting internal states.
        '''
        pass

    def reset_params(self):
        '''
        Utility function for updating parameters which depend on the ros parameter
            server. Setting parameters, such as gains, can be useful for interative
            testing.
        '''
        with self.path_lock:
            self.speed = float(rospy.get_param("/pid/speed", 1.0))
            self.finish_threshold = float(rospy.get_param("/pid/finish_threshold", 0.2))
            self.exceed_threshold = float(rospy.get_param("/pid/exceed_threshold", 4.0))
            # Lookahead distance from current pose to next waypoint. Different from
            # waypoint_lookahead in the other controllers, as those are distance from
            # the reference point.
            self.pose_lookahead = float(rospy.get_param("/purepursuit/pose_lookahead", 0.6))
