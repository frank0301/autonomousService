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
from common_interface.msg import RectDepth,Camera2map
from sensor_msgs.msg import Image,CompressedImage
from std_msgs.msg import Int32MultiArray, Float32MultiArray, String

#import Class and functions from floder "sctipts"
import scripts.task_nav as _task
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
        self.act_list = None
        self.relation_list= None
        # load vlm model, it takes time
        self.VL = GroundingDINOInfer()

        self.memory_builder = MemoryBuilder()
        self.memory_file = "memory.yaml"
        self.update_flag = 1 
        self.current_room = None
        self.room_pose = [0.0, 0.0, 0.0]  # Default pose, should be updated with actual robot pose
        
        self.create_subscription(CompressedImage, '/camera/camera/color/image_raw/compressed', self.rgb_callback, 10)
        self.create_subscription(Image, '/camera/camera/depth/image_rect_raw', self.depth_callback, 10)
        # sub msg from the robot. (test 2 HZ - June 09)
        self.create_subscription(Camera2map, '/camera2map', self.camera2map_callback, 10)
        self.create_subscription(String, '/robot_state',self.robot_state_update_callback, 10)
        
        self.target_pub = self.create_publisher(RectDepth, 'task/rect_depth', 10) # need to be simplify
        self.get_logger().info("ServiceNode node started, waiting for image...")

        self.update_memory_map = self.create_timer(1, self.update_map)
        self.show_img = self.create_timer(1, self.show_rgb)

    def save_to_memory(self, room_type, features_with_coords):
        memory_data = {"nodes": []}
        
        # Load existing memory if it exists
        if os.path.exists(self.memory_file):
            with open(self.memory_file, 'r') as f:
                try:
                    memory_data = yaml.safe_load(f) or {"nodes": []}
                except yaml.YAMLError:
                    self.get_logger().error("Error reading memory file")
                    memory_data = {"nodes": []}
        else:
            # create a new file
            memory_data = {"nodes": []}
            with open(self.memory_file, 'w') as f:
                yaml.safe_dump(memory_data, f)


        # Create new room node
        new_room = {
            "name": room_type,
            "pose": self.room_pose,
            "features": features_with_coords
        }

        # Check if room already exists
        room_exists = False
        for node in memory_data["nodes"]:
            if node["name"] == room_type:
                # Update existing room
                node["features"] = features_with_coords
                room_exists = True
                break

        if not room_exists:
            memory_data["nodes"].append(new_room)

        # Save to file
        with open(self.memory_file, 'w') as f:
            yaml.dump(memory_data, f, default_flow_style=False)
            self.get_logger().info(f"Updated memory file with {room_type} features")

    def update_map(self):
        self.get_logger().info(f"{self.update_flag}")
        if self.update_flag and (self.rgb_image is not None):
            self.update_flag = 0
            # Convert to PIL Image for GPT processing
            pil_image = PILImage.fromarray(cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2RGB))
            # Get map building analysis
            map_analysis = lm.gpt_map_build(pil_image)
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
                            # Calculate world coordinates
                            dis, camer_x, camer_y = self.pix2camera_frame(center)
                            if dis is not None and dis > 0:
                                self.get_logger().info(f"Detected {obj_name} at distance {dis:.2f}m, coordinates ({camer_x:.2f}, {camer_y:.2f})")
                                
                                # Add to features list
                                features_with_coords.append({
                                    "object": obj_name,
                                    "Coordinate relative to the camera frame": [camer_x, camer_y]
                                })
            
                # Save to memory if we have features
                if features_with_coords and "room_type" in map_analysis:
                    self.memory_builder.save_to_memory(map_analysis["room_type"], features_with_coords)
            self.get_logger().info("updated featured in map!")


    def show_rgb(self):
        cv2.imshow("rgb",self.rgb_image)
        cv2.waitKey(1)


    def robot_state_update_callback(self, msg):
        self.robot_state = msg.data
        self.get_logger().info(f"{msg.data}")

    def camera2map_callback(self, msg):
        """Handle camera to map transformation updates"""
        try:
            # Assuming the message contains [x, y, yaw]
            # x, y, yaw = msg.data
            x, y = msg.coordinate.data
            yaw = msg.theta
            # Update the camera pose in memory builder, hook to the memery class
            self.memory_builder.update_camera_pose(x, y, yaw)
            self.get_logger().info(f"Updated camera pose: x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}")
        except Exception as e:
            self.get_logger().error(f"Error processing camera pose: {e}")

    def rgb_callback(self, msg):
        try:
            self.rgb_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.update_flag = 1
            # cv2.imshow("rs_img", self.rgb_image)
            # cv2.waitKey(100)
        except Exception as e:
            self.get_logger().error(f"图像处理失败: {e}")

    def depth_callback(self, msg):
        try:
            # if format of 16UC1, do not use 'passthrough'
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

            depth_normalized = cv2.normalize(self.depth_image, None, 0, 255, cv2.NORM_MINMAX)
            depth_normalized = np.uint8(depth_normalized)  # 转成 8位


        except Exception as e:
            self.get_logger().error(f"Depth image failed to transfer: {e}")

    def pix2camera_frame(self, pix_xy):
        if self.depth_image is None:
            self.get_logger().warn("⚠️ Depth image not yet received.")
            return None, None, None

        center_depth_mm = self.depth_image[pix_xy[1], pix_xy[0]]  # Access depth image using (row, col)
        center_depth_m = center_depth_mm / 1000.0

        self.get_logger().info(f"📏 Center depth at ({pix_xy[0]},{pix_xy[1]}): {center_depth_m:.3f} m")
        cx, cy = 319.47, 247
        fx, fy = 615.53, 615.53
        pix_x, pix_y = pix_xy 
        wY = (pix_x - cx) * center_depth_m / fx
        wX = center_depth_m 
        self.get_logger().info(f"wY:{wY},wX:{wX}")
        return center_depth_m, wX, -wY

    

def main(args=None):
    rclpy.init(args=args)
    node = ServiceNode()

    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    try:
        while True:
            time.sleep(1)
        """
        while True:
            msg = RectDepth()          

            if node.rgb_image is None:
                time.sleep(1)
                continue
            question = input("\n enter your question(type Ctrl+C exit):\n> ").strip()
            if not question:
                print("invalid question")
                continue
            answer = ans2json.ans2json(lm.ask_gpt_ll(question))
            print(f"\n✅ GPT-4o answer: \n{answer}")
            node.act_list, node.obj_list = answer["actions"], answer["objects"]
            print(node.act_list, node.obj_list)
            '''
                    task here: 
                    pub the goal
                        <-
                    detect the obj VLM <- depth img, rect 
            '''
            # node.obj_list = ['a blue trash']
            # node.act_list = ['null']
            idx = 0
            while idx < len(node.act_list):
                obj = node.obj_list[idx]
                act = node.act_list[idx]
                msg = RectDepth()
                if obj and obj.lower() != "null":
                    img_detect, rect, center = node.VL.infer(node.rgb_image, node.obj_list[idx]+".")
                    node.rect = rect
                    # print(rect," ", center)
                    if rect is None:
                        print("no object found")
                        continue
                    cv2.imshow("VL-detect", img_detect)
                    cv2.waitKey(1)
                    dis,wx,wy = node.pix2camera_frame(center)
                    if dis is None:
                        continue
                    # print(f"dis={dis},coordinate=({wx},{wy})")
                    
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
                    else:
                        msg.rect = Int32MultiArray()
                        msg.rect.data = rect
                        msg.center = Int32MultiArray()
                        msg.center.data = center
                        
                        msg.frame = time.time()
                        msg.depth = dis
                        msg.coordinate_diff = Float32MultiArray()

                        msg.coordinate_diff.data = [wx, wy]
                        
                        node.target_pub.publish(msg)
                        time.sleep(3)
                elif act and act.lower() != "null":
                    msg.coodinate_diff = [0, 0]
                    msg.theta = float(act)
                    node.target_pub.publish(msg)
                    print("waiting turning")
                    idx += 1
                    time.sleep(5)
                
                else:
                    node.get_logger().warn("both null, jump over it!")
                    idx+=1
            node.get_logger().info("success!")
    """
    except KeyboardInterrupt:
        print("⛔ 退出程序")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


