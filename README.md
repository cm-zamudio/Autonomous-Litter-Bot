# Autonomous-Litter-Bot

This is our senior design project code for an autonmous litter bot which uses an Openmanipulator-X, Intel realsense camera, and Unitree A1 robot dog. This code is for the autonmous detection of an object, in our case we used the COCO dataset and the object detection is currently only set for bottles. Once a bottle is detected by the intel realsense camera, the Openmanipulator-X (robot arm) will move to the specified XYZ coordinates of the detected object, pick up the object, and then move to a specified XYZ location for dropoff. This code only runs once and not in a loop. We used the Openmanipulator-X ROS2 humble files from the ROBOTIS emanual and the Intel realsense SDK. The project is ran in a docker container on a Raspberry Pi 5 with ROS2 humble installed in  the docker container.

Thank you to andreasHovaldt on Github for sharing his repo containing the orginal yolov8_node.py whch we then altered for the specifcations of our project. Please look at his repo to see his original work. 
https://github.com/andreasHovaldt/yolov8_ros2/tree/humble

Steps to run code commands:

1) Run your docker container
2) ros2 launch open_manipulator_x_bringup hardware.launch.py
3) ros2 launch open_manipulator_x_moveit_config moveit_core.launch.py
4) ros2 launch yolov8_ros2 camera_yolo.launch.py
5) ros2 run open_manipulator_x_controller cartesian_trajectory_executor
