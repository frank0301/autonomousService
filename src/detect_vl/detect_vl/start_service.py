"""
###########################
# Autonomous Navigation Service Node
# LLM: GPT-4o | VLM: Grounding DINO
#
# Main Functions:
# 1. Understand task requirements and extract key features
# 2. Extract targets for VLM, obtain distance from depth map, solve world coordinate offset
# 3. Publish coordinates_diff for navigation
############################
"""

import rclpy
from rclpy.node import Node
import threading
import time
import os

# Vision packages
from cv_bridge import CvBridge
import cv2
from PIL import Image as PILImage
import numpy as np

# ROS2 messages
from common_interface.msg import RectDepth, Camera2map
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Int32MultiArray, Float32MultiArray, String
from std_srvs.srv import Trigger

# Custom modules
from scripts.service_vl import GroundingDINOInfer
import scripts.service_lm as lm
import scripts.ans2json as ans2json
from scripts.memory_builder import MemoryBuilder
import yaml

# Set environment variable for display
os.environ["QT_QPA_PLATFORM"] = "xcb"


class ServiceNode(Node):
    def __init__(self):
        super().__init__('detect_vl_node')
        
        # Initialize core components
        self.bridge = CvBridge()
        self.VL = GroundingDINOInfer()
        self.memory_builder = MemoryBuilder()
        
        # Image data
        self.rgb_image = None
        self.depth_image = None
        
        # Task data
        self.obj_list = None
        self.turn_list = None
        self.relation_list = None
        self.rect: list[int] | None = None
        
        # State variables
        self.robot_state = "unknown"
        self.environment_context = ""
        self.current_room = None
        self.room_pose = [0.0, 0.0, 0.0]
        
        # Control flags
        self.update_flag = 1
        self.suppress_background_activity = False
        
        # Camera2map monitoring
        self.last_camera2map_time = 0
        self.camera2map_warning_threshold = 5.0
        self.camera2map_warning_sent = False
        
        # Display control
        self.last_display_update: float = 0.0
        self.display_update_interval = 0.1
        
        # File paths
        self.memory_file = "/src/memory.yaml"
        
        self._setup_subscriptions_and_publishers()
        self._setup_timers()
        
        self.get_logger().info("ServiceNode started - waiting for images and camera2map messages...")

    def _setup_subscriptions_and_publishers(self):
        """Initialize ROS2 subscriptions and publishers"""
        # Subscriptions
        # self.create_subscription(
        #     CompressedImage, 
        #     '/camera/camera/color/image_raw/compressed', 
        #     self.rgb_callback, 10
        # )
        # self.create_subscription(
        #     Image, 
        #     '/camera/camera/depth/image_rect_raw', 
        #     self.depth_callback, 10
        # )

        #For Simulation only --------------------------------
        self.create_subscription(
            Image, 
            '/camera/camera/color/image_raw', 
            self.rgb_callback, 10
        )
        self.create_subscription(
            Image, 
            '/camera/camera/depth/image_raw', 
            self.depth_callback, 10
        )
          #For Simulation only --------------------------------
          
        self.create_subscription(
            Camera2map, 
            '/camera2map', 
            self.camera2map_callback, 10
        )
        self.create_subscription(
            String, 
            '/robot_state', 
            self.robot_state_update_callback, 10
        )
        
        # Publishers
        self.target_pub = self.create_publisher(RectDepth, 'task/rect_depth', 10)

    def _setup_timers(self):
        """Initialize ROS2 timers"""
        self.update_memory_map = self.create_timer(1.0, self.update_map)
        self.camera2map_monitor_timer = self.create_timer(2.0, self.monitor_camera2map_topic)

    def suppress_background_logging(self, suppress=True):
        """Enable or disable background activity logging"""
        self.suppress_background_activity = suppress
        if suppress:
            self.get_logger().info("🔇 Background activity logging suppressed")
        else:
            self.get_logger().info("🔊 Background activity logging enabled")

    # ============================================================================
    # CALLBACK FUNCTIONS
    # ============================================================================

    def rgb_callback(self, msg):
        """Handle RGB image messages"""
        try:
            self.rgb_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.update_flag = 1
        except Exception as e:
            self.get_logger().error(f"Error processing RGB image: {e}")

    def depth_callback(self, msg):
        """Handle depth image messages"""
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            
            if self.depth_image is not None and self.depth_image.size != 0:
                # Normalize depth for visualization
                depth_normalized = np.zeros_like(self.depth_image)
                cv2.normalize(self.depth_image, depth_normalized, 0, 255, cv2.NORM_MINMAX)
                depth_normalized = np.uint8(depth_normalized)
        except Exception as e:
            self.get_logger().error(f"Depth image failed to transfer: {e}")

    def camera2map_callback(self, msg):
        """Handle camera to map transformation updates"""
        try:
            self.last_camera2map_time = time.time()
            self.camera2map_warning_sent = False
            
            wx, wy, yaw = msg.coordinate.data
            
            if not self.suppress_background_activity:
                self.memory_builder.update_camera_pose(wx, wy, yaw, self.get_logger())
            else:
                self.memory_builder.update_camera_pose(wx, wy, yaw, None)
        except Exception as e:
            self.get_logger().error(f"Error processing camera pose: {e}")

    def robot_state_update_callback(self, msg):
        """Update robot state from navigation node"""
        self.robot_state = msg.data
        if not self.suppress_background_activity:
            self.get_logger().info(f"🤖 Robot state updated to: {self.robot_state}")

    # ============================================================================
    # TIMER FUNCTIONS
    # ============================================================================

    def update_map(self):
        """Update semantic map with detected features"""
        if self.suppress_background_activity or not self.update_flag or self.rgb_image is None:
            return
            
        self.update_flag = 0
        pil_image = PILImage.fromarray(cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2RGB))
        
        try:
            if self.environment_context:
                self.get_logger().info(f"🗺️ Using environment context: {self.environment_context}")
            
            map_analysis = lm.gpt_map_build(pil_image, self.environment_context)
            map_analysis = ans2json.ans2json(map_analysis)
            self.get_logger().info(f"Map Analysis: {map_analysis}")
            
        except Exception as e:
            self.get_logger().error(f"OpenAI API error in map building: {e}")
            self.get_logger().warn("Skipping map update due to API timeout/error")
            return
        
        features_with_coords = self._process_map_features(map_analysis)
        self._save_room_classification(map_analysis, features_with_coords)
        
        self.get_logger().info("Updated features in map!")

    def _process_map_features(self, map_analysis):
        """Process detected features and calculate coordinates"""
        features_with_coords = []
        
        if map_analysis and "features" in map_analysis:
            for feature in map_analysis["features"]:
                if "object" in feature:
                    obj_name = feature["object"]
                    img_detect, rect, center = self.VL.infer(self.rgb_image, obj_name + ".")
                    
                    if rect is not None and center is not None:
                        dis, wx, wy = self.memory_builder.pix2camera_frame(
                            center, self.depth_image, self.get_logger()
                        )
                        
                        if dis is not None and dis > 0:
                            features_with_coords.append({
                                "object": obj_name,
                                "Coordinate relative to the camera frame": [wx, wy]
                            })
        
        return features_with_coords

    def _save_room_classification(self, map_analysis, features_with_coords):
        """Save room classification and features to memory"""
        if not features_with_coords or "room_type" not in map_analysis:
            return
            
        proposed_room_type = map_analysis["room_type"]
        room_pose = self.memory_builder.camera_pose if self.memory_builder.camera_pose else [0.0, 0.0, 0.0]
        
        if self.memory_builder.can_classify_new_room(proposed_room_type, self.get_logger()):
            self.memory_builder.save_to_memory(proposed_room_type, features_with_coords, room_pose)
            self.get_logger().info(f"✅ Room '{proposed_room_type}' classified and saved to memory at pose {room_pose}")
        else:
            if self.memory_builder.last_room_type:
                self.memory_builder.save_to_memory(self.memory_builder.last_room_type, features_with_coords, room_pose)
                self.get_logger().info(f"📝 Features saved to existing room '{self.memory_builder.last_room_type}' (no new room classification)")
            else:
                self.get_logger().warn("⚠️ No room type available for feature storage")



    def monitor_camera2map_topic(self):
        """Monitor camera2map topic and warn if no messages received"""
        if self.suppress_background_activity:
            return
            
        current_time = time.time()
        time_since_last_message = current_time - self.last_camera2map_time
        
        if time_since_last_message > self.camera2map_warning_threshold and not self.camera2map_warning_sent:
            self.get_logger().warn(f"⚠️ No /camera2map messages received for {time_since_last_message:.1f} seconds. Camera pose might not be updated.")
            self.camera2map_warning_sent = True
        elif time_since_last_message <= self.camera2map_warning_threshold and self.camera2map_warning_sent:
            self.get_logger().info("✅ /camera2map topic is now receiving messages again.")
            self.camera2map_warning_sent = False

    # ============================================================================
    # NAVIGATION EXECUTION
    # ============================================================================

    def _process_object_navigation(self, obj, relation, idx):
        """Process navigation to detected object"""
        img_detect, rect, center = self.VL.infer(self.rgb_image, obj + ".")
        self.rect = rect
        
        if img_detect is not None:
            cv2.imshow("VLM Detection", img_detect)
            cv2.waitKey(1)

        if rect is None:
            print("no object found")
            return False
        
        dis, wx, wy = self.memory_builder.pix2camera_frame(center, self.depth_image, self.get_logger())
        if dis is None or dis == 0:
            return False
        
        # Adjust distance based on spatial relation
        dis = self._adjust_distance_by_relation(dis, relation)
        
        # Create and publish navigation message
        msg = self._create_navigation_message(rect, center, dis, wx, wy)
        self.target_pub.publish(msg)
        
        print(f"Goal sent for object '{obj}' at coordinates ({wx:.2f}, {wy:.2f}), distance: {dis:.2f}m")
        return True

    def _adjust_distance_by_relation(self, distance, relation):
        """Adjust target distance based on spatial relation"""
        if relation == 'near':
            return distance - 1.0
        elif relation == 'through':
            return distance + 0.5
        elif relation == 'at':
            return distance
        return distance

    def _create_navigation_message(self, rect, center, dis, wx, wy, theta=0.0):
        """Create RectDepth message for navigation"""
        msg = RectDepth()
        
        msg.rect = Int32MultiArray()
        msg.rect.data = rect
        
        msg.center = Int32MultiArray()
        msg.center.data = center
        
        msg.frame = time.time()
        msg.depth = dis
        msg.theta = theta
        
        msg.coordinate_diff = Float32MultiArray()
        msg.coordinate_diff.data = [wx, wy]
        
        return msg

    def _wait_for_goal_completion(self):
        """Wait for robot to reach the current goal"""
        print("Waiting for robot to reach goal...")
        while self.robot_state != "reachGoal":
            print(f"Robot state: {self.robot_state}")
            time.sleep(0.5)
        print("Robot reached goal!")

    def _process_turn_action(self, act):
        """Process turn action"""
        msg = self._create_navigation_message([], [], 0.0, 0.0, 0.0, float(act))
        msg.coordinate_diff = Float32MultiArray()
        msg.coordinate_diff.data = [0.0, 0.0]
        
        self.target_pub.publish(msg)
        print(f"Turn command: {msg.theta}")
        print("Waiting for turn completion...")
        time.sleep(3)
        
        while self.robot_state != "reachGoal":
            print(f"Robot state: {self.robot_state}")
            time.sleep(0.5)
        print("Turn completed!")


def main(args=None):
    rclpy.init(args=args)
    node = ServiceNode()

    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    
    try:
        while True:
            # Wait for camera data
            if node.rgb_image is None:
                time.sleep(1)
                continue
            
            # Get user input with background activity suppressed
            node.suppress_background_logging(True)
            
            context = input("\n📝 Please provide context about the environment (e.g., Warehouse, Supermarket, etc):\n> ").strip()
            node.environment_context = context
            
            question = input("\nEnter your question (type Ctrl+C to exit):\n> ").strip()
            if not question:
                print("Invalid question")
                continue
                
            node.suppress_background_logging(False)
            
            # Process command with GPT
            answer = ans2json.ans2json(lm.ask_gpt_ll(question))
            print(f"\n✅ GPT-4o answer: \n{answer}")
            
            node.turn_list = answer["turn"]
            node.obj_list = answer["objects"]
            node.relation_list = answer["relative"]
            print(node.obj_list, node.turn_list, node.relation_list)
            
            # Execute navigation sequence
            idx = 0
            goal_sent = False
            
            while idx < len(node.turn_list):
                obj = node.obj_list[idx]
                act = node.turn_list[idx]
                relation = node.relation_list[idx]
                
                if obj and obj.lower() != "null":
                    if not goal_sent:
                        if node._process_object_navigation(obj, relation, idx):
                            goal_sent = True
                        else:
                            continue
                    
                    if node.robot_state == "navigating" or goal_sent:
                        node._wait_for_goal_completion()
                        idx += 1
                        goal_sent = False
                        time.sleep(3)
                        
                elif act and act.lower() != "null":
                    node._process_turn_action(act)
                    idx += 1
                    
                else:
                    node.get_logger().warn("Both object and action are null, skipping...")
                    idx += 1
            
            print(f"Completed {idx} navigation steps")
            node.get_logger().info("Navigation sequence completed successfully!")
            node.robot_state = "reachGoal"
            break

    except KeyboardInterrupt:
        print("⛔ Shutting down due to KeyboardInterrupt")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()