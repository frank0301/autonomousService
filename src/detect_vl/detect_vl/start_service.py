'''
###########################
# use LLM: CHAT-GPT 
#     LVM: Grounding Dino
#
#
this node func: 
1 Understand the task requirements and extract key features.
2 Extract the target for the VLM large model, obtain the distance from the depth map, and solve the world coordinate offset
3 Then publish the coordinates_diff
############################
'''

import rclpy
from rclpy.node import Node

# vision package
from cv_bridge import CvBridge
import cv2
from PIL import Image as PILImage
import numpy as np
# sys package
import os
import threading
import time

#import msgs
from common_interface.msg import RectDepth, Camera2map
from sensor_msgs.msg import Image,CompressedImage
from std_msgs.msg import Int32MultiArray, Float32MultiArray, String
from std_srvs.srv import Trigger

#import Class and functions from floder "sctipts"
from scripts.service_vl import GroundingDINOInfer
import scripts.service_lm as lm
import scripts.ans2json as ans2json
os.environ["QT_QPA_PLATFORM"] = "xcb"

from scripts.memory_builder import MemoryBuilder
import yaml


class ServiceNode(Node):
    def __init__(self):
        super().__init__('detect_vl_node')
        self.bridge = CvBridge()
        
        # images from the camera
        self.rgb_image = None
        self.depth_image = None       
        # task lists
        self.obj_list = None
        self.turn_list = None
        self.relation_list = None
        # load vlm model, it takes time
        self.VL = GroundingDINOInfer()
        # rect attribute for object detection
        self.rect: list[int] | None = None

        self.memory_builder = MemoryBuilder()
        self.memory_file = "/src/memory.yaml"
        self.update_flag = 1 
        self.current_room = None
        self.room_pose = [0.0, 0.0, 0.0]  # Default pose, should be updated with actual robot pose
        
        # Camera2map topic monitoring
        self.last_camera2map_time = 0
        self.camera2map_warning_threshold = 5.0  # seconds - warn if no messages for this long
        self.camera2map_warning_sent = False
        
        # Display control variables to reduce lag
        self.last_display_update: float = 0.0
        self.display_update_interval = 0.1  # Update display every 100ms
        
        # Environment context for map building
        self.environment_context = ""
        
        # Flag to control background activity suppression
        self.suppress_background_activity = False
        
        # Subscribe to compressed RGB image
        self.create_subscription(CompressedImage, '/camera/camera/color/image_raw/compressed', self.rgb_callback, 10)
        
        # Subscribe to depth image (using standard ROS2 approach)
        self.create_subscription(Image, '/camera/camera/depth/image_rect_raw', self.depth_callback, 10)
        
        # sub msg from the robot. (test 2 HZ - June 09)
        self.create_subscription(Camera2map, '/camera2map', self.camera2map_callback, 10)
        self.create_subscription(String, '/robot_state',self.robot_state_update_callback, 10)
        
        self.target_pub = self.create_publisher(RectDepth, 'task/rect_depth', 10)
        self.get_logger().info("ServiceNode node started, waiting for images and camera2map messages...")
        self.update_memory_map = self.create_timer(1, self.update_map)

        # Add door detection timer
        self.door_detection_timer = self.create_timer(0.5, self.detect_doors)
        # Add camera2map monitoring timer
        self.camera2map_monitor_timer = self.create_timer(2.0, self.monitor_camera2map_topic)

        self.robot_state = "unknown"  # Initialize robot state

    def suppress_background_logging(self, suppress=True):
        """Enable or disable background activity logging"""
        self.suppress_background_activity = suppress
        if suppress:
            self.get_logger().info("🔇 Background activity logging suppressed")
        else:
            self.get_logger().info("🔊 Background activity logging enabled")

    def update_map(self):
        # Skip if background activity is suppressed
        if self.suppress_background_activity:
            return
            
        # self.get_logger().info(f"{self.update_flag}")  # Comment out this line
        if self.update_flag and (self.rgb_image is not None):
            self.update_flag = 0
            # Convert to PIL Image for GPT processing
            pil_image = PILImage.fromarray(cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2RGB))
            # Get map building analysis with context
            if self.environment_context:
                self.get_logger().info(f"🗺️ Using environment context: {self.environment_context}")
            map_analysis = lm.gpt_map_build(pil_image, self.environment_context)
            map_analysis = ans2json.ans2json(map_analysis)
            self.get_logger().info(f"Map Analysis: {map_analysis}")
            
            features_with_coords = []
            # Process each feature from the map analysis
            if map_analysis and "features" in map_analysis:
                for feature in map_analysis["features"]:
                    if "object" in feature:
                        obj_name = feature["object"]
                        # Use VLM to detect the object
                        img_detect, rect, center = self.VL.infer(self.rgb_image, obj_name + ".")
                        if rect is not None and center is not None:
                            # Calculate world coordinates using memory_builder
                            dis, wx, wy = self.memory_builder.pix2camera_frame(center, self.depth_image, self.get_logger())
                            if dis is not None and dis > 0:
                                # self.get_logger().info(f"Detected {obj_name} at distance {dis:.2f}m, camera coordinates ({wx:.2f}, {wy:.2f})")
                                
                                # Add to features list with camera frame coordinates
                                features_with_coords.append({
                                    "object": obj_name,
                                    "Coordinate relative to the camera frame": [wx, wy]
                                })
            
            # Check if we can classify a new room type before saving to memory
            if features_with_coords and "room_type" in map_analysis:
                proposed_room_type = map_analysis["room_type"]
                
                # Use camera pose as room pose (if available)
                room_pose = self.memory_builder.camera_pose if self.memory_builder.camera_pose else [0.0, 0.0, 0.0]
                
                if self.memory_builder.can_classify_new_room(proposed_room_type, self.get_logger()):
                    # Room classification allowed, save to memory
                    self.memory_builder.save_to_memory(proposed_room_type, features_with_coords, room_pose)
                    self.get_logger().info(f"✅ Room '{proposed_room_type}' classified and saved to memory at pose {room_pose}")
                else:
                    # Room classification blocked, but still save features to current room
                    if self.memory_builder.last_room_type:
                        self.memory_builder.save_to_memory(self.memory_builder.last_room_type, features_with_coords, room_pose)
                        self.get_logger().info(f"📝 Features saved to existing room '{self.memory_builder.last_room_type}' (no new room classification)")
                    else:
                        # If no previous room type, use a default or skip
                        self.get_logger().warn("⚠️ No room type available for feature storage")
            self.get_logger().info("updated featured in map!")

    def robot_state_update_callback(self, msg):
        self.robot_state = msg.data
        if not self.suppress_background_activity:
            self.get_logger().info(f"{msg.data}")

    def camera2map_callback(self, msg):
        """Handle camera to map transformation updates"""
        try:
            # Record the timestamp of this message
            self.last_camera2map_time = time.time()
            self.camera2map_warning_sent = False  # Reset warning flag since we received a message
            
            # Extract coordinates and yaw from the message
            # coordinate.data contains [wx, wy, yaw] (camera frame coordinates)
            wx, wy, yaw = msg.coordinate.data
            # Update the camera pose in memory builder, hook to the memery class
            if not self.suppress_background_activity:
                self.memory_builder.update_camera_pose(wx, wy, yaw, self.get_logger())
            else:
                self.memory_builder.update_camera_pose(wx, wy, yaw, None)
        except Exception as e:
            self.get_logger().error(f"Error processing camera pose: {e}")

    def rgb_callback(self, msg):
        try:
            self.rgb_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.update_flag = 1
        except Exception as e:
            self.get_logger().error(f"Error processing RGB image: {e}")

    def depth_callback(self, msg):
        """Handle depth image messages"""
        try:
            # Convert depth image to OpenCV format
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

            if self.depth_image is not None and self.depth_image.size != 0:
                # Normalize depth for visualization (optional)
                depth_normalized = np.zeros_like(self.depth_image)
                cv2.normalize(self.depth_image, depth_normalized, 0, 255, cv2.NORM_MINMAX)
                depth_normalized = np.uint8(depth_normalized)
        except Exception as e:
            self.get_logger().error(f"Depth image failed to transfer: {e}")

    def detect_doors(self):
        """Detect doors using memory_builder"""
        if not self.suppress_background_activity:
            self.memory_builder.detect_doors(self.rgb_image, self.depth_image, self.VL, self.get_logger())
        else:
            # Pass None logger to suppress all output
            self.memory_builder.detect_doors(self.rgb_image, self.depth_image, self.VL, None)

    def get_room_transition_status(self):
        """Get current room transition status for debugging"""
        return self.memory_builder.get_room_transition_status()

    def get_camera2map_status(self):
        """Get current camera2map topic status"""
        current_time = time.time()
        time_since_last_message = current_time - self.last_camera2map_time
        
        if time_since_last_message > self.camera2map_warning_threshold:
            return {
                "status": "warning",
                "time_since_last_message": time_since_last_message,
                "message": f"No /camera2map messages received for {time_since_last_message:.1f} seconds"
            }
        else:
            return {
                "status": "normal",
                "time_since_last_message": time_since_last_message,
                "message": f"Last message received {time_since_last_message:.1f} seconds ago"
            }

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


def main(args=None):
    rclpy.init(args=args)
    node = ServiceNode()

    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    try:
        # while True:
        #     time.sleep(1)

        while True:
            msg = RectDepth()          

            if node.rgb_image is None:
                time.sleep(1)
                continue

            
           # Suppress background activity before asking for user input
            node.suppress_background_logging(True)
            
            # Ask for environment context first
            context = input("\n📝 Please provide context about the environment (e.g.,Warehouse, Supermarket, etc)':\n> ").strip()
            node.environment_context = context  # Store the context for map building
            
            question = input("\n enter your question(type Ctrl+C exit):\n> ").strip()
            if not question:
                print("invalid question")
                continue
                
            # Re-enable background activity after user input is complete
            node.suppress_background_logging(False)
            
            answer = ans2json.ans2json(lm.ask_gpt_ll(question))
            print(f"\n✅ GPT-4o answer: \n{answer}")
            node.turn_list, node.obj_list,node.relation_list = answer["turn"], answer["objects"], answer["relative"]
            print(node.obj_list, node.turn_list, node.relation_list)
            '''
                    task here: 
                    pub the goal
                        <-
                    detect the obj VLM <- depth img, rect 
            '''
            # node.obj_list = ['a blue trash']
            # node.turn_list = ['null']
            idx = 0
            while idx < len(node.turn_list):
                obj = node.obj_list[idx]
                act = node.turn_list[idx]
                relation = node.relation_list[idx]
                msg = RectDepth()
                if obj and obj.lower() != "null":
                    img_detect, rect, center = node.VL.infer(node.rgb_image, node.obj_list[idx]+".")
                    node.rect = rect
                    
                    # Display VLM detection results in real-time
                    if img_detect is not None:
                        cv2.imshow("VLM Detection", img_detect)
                        cv2.waitKey(1)  # Update display
                    
                    # print(rect," ", center)
                    if rect is None:
                        print("no object found")
                        continue
                    
                    dis,wx,wy = node.memory_builder.pix2camera_frame(center, node.depth_image, node.get_logger())
                    if dis is None:
                        continue
                    
                    if dis == 0:
                        continue
                    
                    if dis < 3 and node.robot_state == "navigating":                      
                        print("waitting reachGoal")
                        while node.robot_state != "reachGoal":
                            print(node.robot_state)
                            # rclpy.spin_once(node)
                            time.sleep(0.5)
                        print("reached goal!")
                        idx += 1
                        time.sleep(3)
                    else:
                        if relation == 'near':
                            dis = dis - 1.0
                        elif relation =='through':
                            dis = dis + 0.5
                        elif relation == 'at':
                            dis = dis
                        # elif relation == 'toward':
                        #     dis = 
                        
                        msg.rect = Int32MultiArray()
                        msg.rect.data = rect

                        msg.center = Int32MultiArray()
                        msg.center.data = center
                        
                        msg.frame = time.time()

                        msg.depth = dis
                        msg.coordinate_diff = Float32MultiArray()
                        msg.coordinate_diff.data = [wx, wy]
                        
                        node.target_pub.publish(msg)
                        print("update target!")
                        time.sleep(3)
                elif act and act.lower() != "null":
                    msg.theta=float(act)
                    print(msg.theta)
                    msg.coordinate_diff = Float32MultiArray()
                    msg.coordinate_diff.data = [0.0, 0.0]
                    # msg.theta = float(act)
                    node.target_pub.publish(msg)
                    # while node.robot_state != "navigating":
                    #     node.target_pub.publish(msg)
                    #     # rclpy.spin_once(node)
                    #     time.sleep(10)
                    print("waiting turning")
                    time.sleep(3)            
                    while node.robot_state != "reachGoal":
                        print(node.robot_state)
                        # rclpy.spin_once(node)
                        time.sleep(0.5)
                    print("reached goal!")
                    idx += 1                             
                else:
                    node.get_logger().warn("both null, jump over it!")
                    idx+=1
            print(idx)
            node.get_logger().info("success!")
            break

    except KeyboardInterrupt:
        print("⛔ ERROR: KeyboardInterrupt")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()