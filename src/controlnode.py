import rospy
import threading

from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseStamped, PoseArray, Pose, PointStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Empty
from std_msgs.msg import Header, Float32
from std_srvs.srv import Empty as SrvEmpty
from mushr_rhc.msg import XYHVPath
from visualization_msgs.msg import Marker

import mpc
import nonlinear
import pid
import purepursuit
import utils

controllers = {
    "PID": pid.PIDController,
    "PP": purepursuit.PurePursuitController,
    "NL": nonlinear.NonLinearController,
    "MPC": mpc.ModelPredictiveController,
}


class ControlNode:
    def __init__(self, name):
        self.ackermann_msg_id = 0        
        self.path_event = threading.Event()
        self.reset_lock = threading.Lock()
        self.ready_event = threading.Event()
        self.start(name)

    def start(self, name):
        rospy.init_node(name, anonymous=True, disable_signals=True)

        self.setup_pub_sub()
        self.load_controller()
        self.ready_event.set()

        rate = rospy.Rate(50)
        self.inferred_pose = None
        print("Control Node Initialized")

        while not rospy.is_shutdown():
            self.path_event.wait()
            self.reset_lock.acquire()
            ip = self.inferred_pose

            if ip is not None and self.controller.ready():
                index = self.controller.get_reference_index(ip)
                pose = self.controller.get_reference_pose(index)
                error = self.controller.get_error(ip, index)
                cte = error[1]

                self.publish_selected_pose(pose)
                self.publish_cte(cte)
                next_ctrl = self.controller.get_control(ip, index)
                if next_ctrl is not None:
                    self.publish_ctrl(next_ctrl)
                if self.controller.path_complete(ip, error):
                    self.path_event.clear()
                    print(ip, error)

            self.reset_lock.release()
            rate.sleep()

    def shutdown(self):
        rospy.signal_shutdown("shutting down from signal")
        self.path_event.clear()
        self.ready_event.clear()
        exit(0)

    def load_controller(self):
        self.controller_type = rospy.get_param("~controller/type", default="MPC")
        print(self.controller_type)
        self.controller = controllers[self.controller_type]()

    def setup_pub_sub(self):
        rospy.Service("~reset/hard", SrvEmpty, self.srv_reset_hard)
        rospy.Service("~reset/state", SrvEmpty,  self.srv_reset_state)
        rospy.Service("~reset/params", SrvEmpty, self.srv_reset_params)

        rospy.Subscriber("/initialpose",
                PoseWithCovarianceStamped, self.cb_init_pose, queue_size=1)

        rospy.Subscriber(
            "/clicked_point",
            PointStamped,
            self.clicked_point_cb,
            queue_size=1
        )

        rospy.Subscriber(
            "/move_base_simple/goal", PoseStamped, self.cb_goal, queue_size=1
        )

        rospy.Subscriber("/controller/set_path",
                XYHVPath, self.cb_path, queue_size=1)

        rospy.Subscriber("/car/particle_filter/inferred_pose",
                         PoseStamped, self.cb_pose, queue_size=10)

        self.rp_ctrls = rospy.Publisher(
            "/car/mux/ackermann_cmd_mux/input/navigation",
            AckermannDriveStamped, queue_size=2
        )

        self.rp_cte = rospy.Publisher(
            rospy.get_param(
                "~cte_viz_topic",
                default="/controller/cte"
            ),
            Float32, queue_size=2
        )

        self.rp_waypoints = rospy.Publisher(
            rospy.get_param(
                "~waypoint_viz_topic",
                default="/controller/path/waypoints"
            ),
            Marker, queue_size=2
        )

        self.rp_waypoint = rospy.Publisher(
            rospy.get_param(
                "~selected_waypoint_viz_topic",
                default="/controller/path/selected_waypoint"
            ),
            PoseStamped, queue_size=2
        )

        self.rp_path_viz = rospy.Publisher(
            rospy.get_param(
                "~poses_viz_topic",
                default="/controller/path/poses"
            ),
            PoseArray, queue_size=2
        )

    def srv_reset_hard(self, msg):
        '''
        Hard reset does a complete reload of the controller.
        '''
        rospy.loginfo("Start hard reset")
        self.reset_lock.acquire()
        self.load_controller()
        self.reset_lock.release()
        rospy.loginfo("End hard reset")
        return []

    def srv_reset_params(self, msg):
        '''
        Param reset resets parameters of the controller. Useful for iterative tuning.
        '''
        rospy.loginfo("Start param reset")
        self.reset_lock.acquire()
        self.controller.reset_params()
        self.reset_lock.release()
        rospy.loginfo("End param reset")
        return []

    def srv_reset_state(self, msg):
        '''
        State reset resets state dependent variables, such as accumulators in PID control.
        '''
        rospy.loginfo("Start state reset")
        self.reset_lock.acquire()
        self.controller.reset_state()
        self.reset_lock.release()
        rospy.loginfo("End state reset")
        return []

    def cb_odom(self, msg):
        self.inferred_pose = utils.rospose_to_posetup(msg.pose.pose)

    def cb_path(self, msg):
        print("Got path!")
        path = msg.path.waypoints
        self.visualize_path(path)
        self.controller.set_path(path)
        self.path_event.set()
        print("Path set")
        return True

    def cb_goal(self, msg):
        self.path = None
        goal = utils.rospose_to_posetup(msg.pose)
        self.controller.set_goal(goal)
        self.path_event.set()
        print("goal set", goal)

    def clicked_point_cb(self, msg):
        self.path = None
        goal = utils.rospoint_to_posetup(msg)
        self.controller.set_goal(goal)
        self.path_event.set()
        print("goal set", goal)

    def cb_pose(self, msg):
        self.inferred_pose = [
            msg.pose.position.x,
            msg.pose.position.y,
            utils.rosquaternion_to_angle(msg.pose.orientation)]

    def publish_ctrl(self, ctrl):
        assert len(ctrl) == 2
        ctrlmsg = AckermannDriveStamped()
        ctrlmsg.header.stamp = rospy.Time.now()
        ctrlmsg.header.seq = self.ackermann_msg_id
        ctrlmsg.drive.speed = ctrl[0]
        ctrlmsg.drive.steering_angle = ctrl[1]
        self.rp_ctrls.publish(ctrlmsg)
        self.ackermann_msg_id += 1

    def visualize_path(self, path):
        marker = self.make_marker(path[0], 0, "start")
        self.rp_waypoints.publish(marker)
        poses = []
        for i in range(1, len(path)):
            p = Pose()
            p.position.x = path[i].x
            p.position.y = path[i].y
            p.orientation = utils.angle_to_rosquaternion(path[i].h)
            poses.append(p)
        pa = PoseArray()
        pa.header = Header()
        pa.header.stamp = rospy.Time.now()
        pa.header.frame_id = "map"
        pa.poses = poses
        self.rp_path_viz.publish(pa)

    def make_marker(self, config, i, point_type):
        marker = Marker()
        marker.header = Header()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = "map"
        marker.ns = str(config)
        marker.id = i
        marker.type = Marker.CUBE
        marker.pose.position.x = config.x
        marker.pose.position.y = config.y
        marker.pose.orientation.w = 1
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.a = 1.0
        if point_type == "waypoint":
            marker.color.b = 1.0
        else:
            marker.color.g = 1.0

        return marker

    def publish_selected_pose(self, pose):
        p = PoseStamped()
        p.header = Header()
        p.header.stamp = rospy.Time.now()
        p.header.frame_id = "map"
        p.pose.position.x = pose[0]
        p.pose.position.y = pose[1]
        p.pose.orientation = utils.angle_to_rosquaternion(pose[2])
        self.rp_waypoint.publish(p)

    def publish_cte(self, cte):
        self.rp_cte.publish(Float32(cte))

    def cb_init_pose(self, pose):
        self.path_event.set()
