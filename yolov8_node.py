# Basic ROS2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import ReliabilityPolicy, QoSProfile


# Executor and callback imports
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor

# ROS2 interfaces
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String

# Image msg parser
from cv_bridge import CvBridge

# Vision model
from ultralytics import YOLO

# Others
import numpy as np
import open3d as o3d
import time, json, torch


class Yolov8Node(Node):
    
    def __init__(self):
        super().__init__("yolov8_node")
        rclpy.logging.set_logger_level('yolov8_node', rclpy.logging.LoggingSeverity.INFO)
        
        ## Declare parameters for node
        self.declare_parameter("model", "yolov8n.pt")  # Changed default to detection model
        model = self.get_parameter("model").get_parameter_value().string_value
        
        self.declare_parameter("device", "cuda:0")
        self.device = self.get_parameter("device").get_parameter_value().string_value
        
        self.declare_parameter("depth_threshold", 1.2)
        self.depth_threshold = self.get_parameter("depth_threshold").get_parameter_value().double_value
        
        self.declare_parameter("threshold", 0.6)
        self.threshold = self.get_parameter("threshold").get_parameter_value().double_value
        
        self.declare_parameter("enable_yolo", True)
        self.enable_yolo = self.get_parameter("enable_yolo").get_parameter_value().bool_value
        
        # Transformations (kept for future use if needed)
        self.tf_world_to_camera = np.array([[-0.000, -1.000,  0.000, -0.017], [0.559,  0.000,  0.829, -0.272], [-0.829,  0.000,  0.559,  0.725], [0.000,  0.000,  0.000,  1.000]])
        self.tf_camera_to_optical = np.array([[-0.003,  0.001,  1.000,  0.000], [-1.000, -0.002, -0.003,  0.015], [0.002, -1.000,  0.001, -0.000], [0.000,  0.000,  0.000,  1.000]])
        self.tf_world_to_optical = np.matmul(self.tf_world_to_camera, self.tf_camera_to_optical)

        
        ## other inits
        self.group_1 = MutuallyExclusiveCallbackGroup() # camera subscribers
        self.group_2 = MutuallyExclusiveCallbackGroup() # vision timer
        
        self.cv_bridge = CvBridge()
        self.yolo = YOLO(model)
        self.yolo.fuse() # Conv2d and BatchNorm2d Layer Fusion for improved performance
        
        self.color_image_msg = None
        self.depth_image_msg = None
        self.camera_intrinsics = None
        self.pred_image_msg = Image()
        
        # Set clipping distance for background removal
        depth_scale = 0.001
        self.depth_threshold = self.depth_threshold/depth_scale
        
        
        # Publishers
        self._item_dict_pub = self.create_publisher(String, "/yolo/prediction/item_dict", 10)
        self._pred_pub = self.create_publisher(Image, "/yolo/prediction/image", 10)
        
        # Subscribers
        self._color_image_sub = self.create_subscription(Image, "/camera/camera/color/image_raw", self.color_image_callback, qos_profile_sensor_data, callback_group=self.group_1)
        self._depth_image_sub = self.create_subscription(Image, "/camera/camera/aligned_depth_to_color/image_raw", self.depth_image_callback, qos_profile_sensor_data, callback_group=self.group_1)
        self._camera_info_subscriber = self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.camera_info_callback, QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE), callback_group=self.group_1)

        # Timers
        self._vision_timer = self.create_timer(0.04, self.object_detection, callback_group=self.group_2) # 25 hz

    
    def color_image_callback(self, msg):
        self.color_image_msg = msg
        
    def depth_image_callback(self, msg):
        self.depth_image_msg = msg
    
    def camera_info_callback(self, msg):
        try:
            if self.camera_intrinsics is None:
                # Set intrinsics in o3d object
                self.camera_intrinsics = o3d.camera.PinholeCameraIntrinsic()
                self.camera_intrinsics.set_intrinsics(msg.width,    #msg.width
                                                  msg.height,       #msg.height
                                                  msg.k[0],         #msg.K[0] -> fx
                                                  msg.k[4],         #msg.K[4] -> fy
                                                  msg.k[2],         #msg.K[2] -> cx
                                                  msg.k[5] )        #msg.K[5] -> cy
                self.get_logger().info('Camera intrinsics have been set!')
            
        except Exception as e:
            self.get_logger().error(f'camera_info_callback Error: {e}')


    def bg_removal(self, color_img_msg: Image, depth_img_msg: Image):
        if self.color_image_msg is not None and self.depth_image_msg is not None:
        
            # Convert color image msg
            cv_color_image = self.cv_bridge.imgmsg_to_cv2(color_img_msg, desired_encoding='bgr8')
            np_color_image = np.array(cv_color_image, dtype=np.uint8)

            # Convert depth image msg
            cv_depth_image = self.cv_bridge.imgmsg_to_cv2(depth_img_msg, desired_encoding='passthrough')
            np_depth_image = np.array(cv_depth_image, dtype=np.uint16)

            # bg removal
            grey_color = 153
            depth_image_3d = np.dstack((np_depth_image, np_depth_image, np_depth_image)) # depth image is 1 channel, color is 3 channels
            bg_removed = np.where((depth_image_3d > self.depth_threshold) | (depth_image_3d != depth_image_3d), grey_color, np_color_image)
            
            return bg_removed, np_color_image, np_depth_image
        
        self.get_logger().error("Background removal error, color or depth msg was None")
        return None, None, None
    
    
    def object_detection(self):
        if not self.enable_yolo or self.color_image_msg is None or self.depth_image_msg is None or self.camera_intrinsics is None:
            return

        self.get_logger().debug("Successfully acquired color and depth image msgs")

        try:
            # Remove background
            bg_removed, np_color_image, np_depth_image = self.bg_removal(self.color_image_msg, self.depth_image_msg)
            if bg_removed is None:
                return
        except Exception as e:
            self.get_logger().error(f"Error during background removal: {str(e)}")
            return
            
        self.get_logger().debug("Successfully removed background")
        
        # Predict on image "bg_removed"
        results = self.yolo.predict(
            source=bg_removed,
            show=False,
            verbose=False,
            stream=False,
            conf=self.threshold,
            device=self.device,
            classes=[39]
        )
        self.get_logger().debug("Successfully predicted")
        
        # Go through detections in prediction results
        for detection in results:
            try:
                pred_img = detection.plot()
                self.pred_image_msg = self.cv_bridge.cv2_to_imgmsg(pred_img, encoding='passthrough')
                self._pred_pub.publish(self.pred_image_msg)
            except Exception as e:
                self.get_logger().error(f"Error plotting detection: {str(e)}")
                continue

            try:
                detection_class = detection.boxes.cls.cpu().numpy()
                detection_conf = detection.boxes.conf.cpu().numpy()
                n_objects = len(detection_class)

                if n_objects == 0:
                    self.get_logger().debug("No objects detected")
                    empty_dict_msg = String()
                    empty_dict_msg.data = json.dumps({})
                    self._item_dict_pub.publish(empty_dict_msg)
                    continue

                object_boxes = detection.boxes.xyxy.cpu().numpy()

                fx = self.camera_intrinsics.get_focal_length()[0]
                fy = self.camera_intrinsics.get_focal_length()[1]
                cx = self.camera_intrinsics.get_principal_point()[0]
                cy = self.camera_intrinsics.get_principal_point()[1]

                item_dict = {}
                for n in range(n_objects):
                    x1, y1, x2, y2 = object_boxes[n].astype(int)
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(np_depth_image.shape[1] - 1, x2)
                    y2 = min(np_depth_image.shape[0] - 1, y2)

                    # Calculate width in pixels and convert to meters using fx and depth
                    # Convert x1 and x2 (pixel columns) to real-world X coordinates using camera intrinsics
                    object_width_px = x2 - x1


                    depth_roi = np_depth_image[y1:y2, x1:x2]
                    valid_depths = depth_roi[depth_roi > 0]
                    if len(valid_depths) == 0:
                        center_xyz = [0, 0, 0]
                        width_m = 0.0
                    else:
                        median_depth = np.median(valid_depths).astype(float)
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        depth_meters = median_depth * 0.001
                        yaxis1 = (center_x - cx) * depth_meters / fx
                        zaxis1 = (center_y - cy) * depth_meters / fy
                        x = depth_meters + 0.050 # Adjusted for gripper position

                        ycorr = -yaxis1 + 0.020 #0.020 offset adjust to center of object
                        zcorr= -zaxis1 + 0.050 #0.050 offset
                        center_xyz = [x, ycorr, zcorr]

                         # Convert width in pixels to meters (approximate)
                        x1_m = (x1 - cx) * depth_meters / fx
                        x2_m = (x2 - cx) * depth_meters / fx
                        width_m = abs(x2_m - x1_m)

                        width_m = width_m / 2 - 0.008 #adjust to gripper width

                    item_dict[f'item_{n}'] = {
                        'class': detection.names[detection_class[n]],
                        'confidence': round(float(detection_conf[n]), 3),
                        #'bbox': [round(float(coord), 3) for coord in object_boxes[n]],
                        'position_xyz': [round(val, 3) for val in center_xyz],
                        'estimated_width_m': round(width_m, 4)

                    }

                self.item_dict = item_dict
                self.item_dict_str = json.dumps(self.item_dict)
                class_names = [detection.names[item] for item in detection_class]
                self.get_logger().debug(f"Yolo detected items: {class_names}")

                item_dict_msg = String()
                item_dict_msg.data = self.item_dict_str
                self._item_dict_pub.publish(item_dict_msg)

                self.get_logger().debug("Item dictionary successfully created and published")

            except Exception as e:
                self.get_logger().error(f"Error processing detection results: {str(e)}")
                continue

            except Exception as e:
                self.get_logger().error(f"Error in object detection: {str(e)}")

            
    
    def shutdown_callback(self):
        self.get_logger().warn("Shutting down...")
        
        

def main(args=None):
    rclpy.init(args=args)

    # Instansiate node class
    vision_node = Yolov8Node()

    # Create executor
    executor = MultiThreadedExecutor()
    executor.add_node(vision_node)

    
    try:
        # Run executor
        executor.spin()
        
    except KeyboardInterrupt:
        pass
    
    finally:
        # Shutdown executor
        vision_node.shutdown_callback()
        executor.shutdown()




if __name__ == "__main__":
    main()
