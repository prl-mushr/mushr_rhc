[![Build Status](https://dev.azure.com/prl-mushr/mushr_rhc/_apis/build/status/prl-mushr.mushr_rhc?branchName=master)](https://dev.azure.com/prl-mushr/mushr_rhc/_build/latest?definitionId=1&branchName=master)

# Receding Horizon Control

This module hosts the RHC controller first implemented on MuSHR stack. It is a model predictive contoller that plans to waypoints from a goal (instead of a reference trajectory). This controller is suitable for cars that don't have a planning module, but want simple MPC.

## Installing on the car
**Note:** if you are using the mushr image you can just clone the repo into `~/catkin_ws/src` and it should work out of the box

Get pip:
```
sudo apt install python-pip
```
To run this module on the car, you need a few packages. To get them download the wheel file for torch from nvidia:
```
$ wget https://nvidia.box.com/shared/static/m6vy0c7rs8t1alrt9dqf7yt1z587d1jk.whl -O torch-1.1.0a0+b457266-cp27-cp27mu-linux_aarch64.whl
$ pip install torch-1.1.0a0+b457266-cp27-cp27mu-linux_aarch64.whl
```
Then get the future package:
```
pip install future
```
Then get the python packages necessary:
```
$ sudo apt install python-scipy
$ sudo apt install python-networkx
$ sudo apt install python-sklearn
```

## `librhc` Layout
`librhc` (`mushr_rhc_ros/src/librhc`) is the core MPC code, with the other source being ROS interfacing code. The main components are:
- Cost function (`librhc/cost`): Takes into account the cost-to-go, collisions and other information to produce a cost for a set of trajectories.
- Model (`librhc/model`): A model of the car, currenly using the kinematic bicycle model.
- Trajectory generation (`librhc/trajgen`): Strategies for generating trajectory libraries for MPC to evaluate.
- Value function (`librhc/value`): Evaluation of positions of the car with resepct to a goal.
- World Representation (`librhc/workrep`): An occupancy grid based representation for the map.

## `mushr_rhc_ros` ROS API

#### Publishers
Topic | Type | Description
------|------|------------
`/rhcontroller/markers`|[visualization_msgs/Marker](http://docs.ros.org/api/visualization_msgs/html/msg/Marker.html)|Halton points sampled in the map (for debugging purposes).
`/rhcontroller/traj_chosen`|[geometry_msgs/PoseArray](http://docs.ros.org/api/geometry_msgs/html/msg/PoseArray.html)|The lowest cost trajectory (for debuggin purposes).
`/mux/ackermann_cmd_mux/input/navigation`|[ackermann_msgs/AckermannDriveStamped](http://docs.ros.org/api/ackermann_msgs/html/msg/AckermannDriveStamped.html)|The lowest cost control to apply on the car.

#### Subscribers
Topic | Type | Description
------|------|------------
`/map_metadata`|[nav_msgs/MapMetaData](http://docs.ros.org/api/nav_msgs/html/msg/MapMetaData.html)|Uses dimension and resolution to create occupancy grid.
`/move_base_simple/goal`|[geometry_msgs/PoseStamped](http://docs.ros.org/api/geometry_msgs/html/msg/PoseStamped.html)|Goal to compute path to.
`/car_pose`|[geometry_msgs/PoseStamped](http://docs.ros.org/api/geometry_msgs/html/msg/PoseStamped.html)|*When using simulated car pose* Current pose of the car.
`/pf/inferred_pose`|[geometry_msgs/PoseStamped](http://docs.ros.org/api/geometry_msgs/html/msg/PoseStamped.html)|*When using particle filter for localization* Current pose of the car.

#### Services
Topic | Type | Description
------|------|------------
`/rhcontroller/reset/hard`|[std_srvs/Empty](http://docs.ros.org/api/std_srvs/html/srv/Empty.html)|Creates a new instance of the MPC object, redoing all initialization computation.
`/rhcontroller/reset/soft`|[std_srvs/Empty](http://docs.ros.org/api/std_srvs/html/srv/Empty.html)|Resets parameters only, not redoing initialization.
