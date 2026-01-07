# Unitree A1 Autonomous-Litter-Bot

Sensior design project Fall 2024 - Spring 2025, University of Illinois Chicago

This project demonstrates a one-shot autonomous pick-and-place pipeline using an OpenManipulator-X mounted on a Unitree A1. An Intel RealSense camera performs object detection (COCO dataset, bottle class only). Once a bottle is detected, its 3D position is computed and the robot arm moves to the object, grasps it, and places it at a predefined drop-off location.

The system runs once per execution (no continuous loop) and is built on ROS 2 Humble, using ROBOTIS OpenManipulator-X packages and the Intel RealSense SDK. All components are containerized and executed in Docker on a Raspberry Pi 5.

Thank you to professor Pranav Bhounsule for being our mentor during the project.

Project Demo: https://youtu.be/8s1aPp4qPkk?si=bJI22zxjL7WrrPki

![image001](https://github.com/user-attachments/assets/250de2b3-7ba9-499d-927b-12c8b718f9e0)

Thank you to andreasHovaldt on Github for sharing his repo containing the orginal yolov8_node.py whch we then altered for the specifcations of our project. Please look at his repo to see his original work. 
https://github.com/andreasHovaldt/yolov8_ros2/tree/humble

Steps to run code commands:

1) Run your docker container
2) ros2 launch open_manipulator_x_bringup hardware.launch.py
3) ros2 launch open_manipulator_x_moveit_config moveit_core.launch.py
4) ros2 launch yolov8_ros2 camera_yolo.launch.py
5) ros2 run open_manipulator_x_controller cartesian_trajectory_executor
