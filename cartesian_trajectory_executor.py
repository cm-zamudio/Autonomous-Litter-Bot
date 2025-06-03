import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose
from moveit_msgs.srv import GetCartesianPath
from moveit_msgs.action import ExecuteTrajectory
from control_msgs.action import GripperCommand
from moveit_msgs.msg import RobotState
from sensor_msgs.msg import JointState
from rclpy.action import ActionClient
import json
import numpy as np

class CartesianTrajectoryExecutor(Node):
    def __init__(self):
        super().__init__('cartesian_trajectory_executor')

        # Clients
        self.compute_cartesian_path_client = self.create_client(GetCartesianPath, '/compute_cartesian_path')
        self.execute_trajectory_client = ActionClient(self, ExecuteTrajectory, '/execute_trajectory')
        self.gripper_client = ActionClient(self, GripperCommand, '/gripper_controller/gripper_cmd')

        # Subscriber
        self.subscription = self.create_subscription(
            String,
            '/yolo/prediction/item_dict',
            self.yolo_callback,
            10)

        # State variables
        self._received_item = False
        self._already_processed = False
        self._target_pose = Pose()
        self._estimated_width_m = 0.019  # Default width

        # Define Home Pose (MoveIt Home)
        self._home_pose = Pose()
        self._home_pose.position.x = 0.2
        self._home_pose.position.y = 0.0
        self._home_pose.position.z = 0.2
        q = eul2quat(0.0, 0.0, 0.0)
        self._home_pose.orientation.x = q[0]
        self._home_pose.orientation.y = q[1]
        self._home_pose.orientation.z = q[2]
        self._home_pose.orientation.w = q[3]

        # Define Drop-off Pose
        self._dropoff_pose = Pose()
        self._dropoff_pose.position.x = 0.2
        self._dropoff_pose.position.y = -0.050
        self._dropoff_pose.position.z = 0.2
        q_drop = eul2quat(0.0, 0.0, 0.0)
        self._dropoff_pose.orientation.x = q_drop[0]
        self._dropoff_pose.orientation.y = q_drop[1]
        self._dropoff_pose.orientation.z = q_drop[2]
        self._dropoff_pose.orientation.w = q_drop[3]

    def yolo_callback(self, msg):
        if self._already_processed:
            return
        
        try:
            item_dict = json.loads(msg.data)
            if item_dict:
                first_item = next(iter(item_dict.values()))
                position_xyz = first_item['position_xyz']
                estimated_width_m = first_item['estimated_width_m']
                
                self.get_logger().info(f"Received target: {position_xyz}, width: {estimated_width_m}")

                self._target_pose.position.x = position_xyz[0]
                self._target_pose.position.y = position_xyz[1]
                self._target_pose.position.z = position_xyz[2]

                q = eul2quat(0.0, 0.0, 0.0)
                self._target_pose.orientation.x = q[0]
                self._target_pose.orientation.y = q[1]
                self._target_pose.orientation.z = q[2]
                self._target_pose.orientation.w = q[3]

                self._estimated_width_m = estimated_width_m
                self._received_item = True

        except Exception as e:
            self.get_logger().error(f"Failed to parse YOLO item_dict: {str(e)}")

    def compute_cartesian_trajectory(self, target_pose):
        while not self.compute_cartesian_path_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /compute_cartesian_path service...')

        request = GetCartesianPath.Request()
        request.header.frame_id = 'link1'
        request.start_state = RobotState()
        request.start_state.joint_state = JointState()
        request.group_name = 'arm'
        request.waypoints.append(target_pose)
        request.max_step = 0.01
        request.jump_threshold = 0.0
        request.avoid_collisions = True

        future = self.compute_cartesian_path_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            return future.result().solution
        else:
            self.get_logger().error('Failed to compute Cartesian path')
            return None

    def execute_trajectory(self, trajectory):
        goal_msg = ExecuteTrajectory.Goal()
        goal_msg.trajectory = trajectory

        self.execute_trajectory_client.wait_for_server()

        send_goal_future = self.execute_trajectory_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_goal_future)

        goal_handle = send_goal_future.result()

        if not goal_handle.accepted:
            self.get_logger().error('Trajectory execution rejected')
            return False

        self.get_logger().info('Trajectory execution accepted, waiting for result...')

        get_result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, get_result_future)

        result = get_result_future.result().result

        if result.error_code.val == 1:
            self.get_logger().info('Trajectory execution completed successfully')
            return True
        else:
            self.get_logger().error(f'Trajectory execution failed with error code: {result.error_code.val}')
            return False

    def send_gripper_command(self, width):
        self.gripper_client.wait_for_server()

        goal_msg = GripperCommand.Goal()
        goal_msg.command.position = width
        goal_msg.command.max_effort = 1.0

        future = self.gripper_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            self.get_logger().info('Gripper command sent successfully')
        else:
            self.get_logger().error('Failed to send gripper command')

def eul2quat(roll_deg: float, pitch_deg: float, yaw_deg: float):
    roll = np.radians(roll_deg)
    pitch = np.radians(pitch_deg)
    yaw = np.radians(yaw_deg)
    cr, sr = np.cos(roll/2), np.sin(roll/2)
    cp, sp = np.cos(pitch/2), np.sin(pitch/2)
    cy, sy = np.cos(yaw/2), np.sin(yaw/2)

    return [
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy,
        cr*cp*cy + sr*sp*sy
    ]

def main(args=None):
    rclpy.init(args=args)
    node = CartesianTrajectoryExecutor()

    while rclpy.ok():
        rclpy.spin_once(node)

        if node._received_item and not node._already_processed:
            # Move to object
            trajectory_to_object = node.compute_cartesian_trajectory(node._target_pose)
            if trajectory_to_object:
                success = node.execute_trajectory(trajectory_to_object)

                if success:
                    # Close gripper
                    node.send_gripper_command(node._estimated_width_m)

                    # Return to Home
                    trajectory_home = node.compute_cartesian_trajectory(node._home_pose)
                    if trajectory_home:
                        node.execute_trajectory(trajectory_home)

                        # Move to Drop-off
                        trajectory_dropoff = node.compute_cartesian_trajectory(node._dropoff_pose)
                        if trajectory_dropoff:
                            success_drop = node.execute_trajectory(trajectory_dropoff)

                            if success_drop:
                                # Open gripper fully at dropoff
                                node.send_gripper_command(0.019)  # Fully open for OpenManipulator X

                                # Return again to Home
                                trajectory_home2 = node.compute_cartesian_trajectory(node._home_pose)
                                if trajectory_home2:
                                    node.execute_trajectory(trajectory_home2)

                    node.get_logger().info("Completed pickup, drop-off, and return to Home!")
                    node._already_processed = True  # lock

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
