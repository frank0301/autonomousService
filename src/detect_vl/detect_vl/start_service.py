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

from sensor_msgs.msg import Image,CompressedImage
from std_msgs.msg import Int32MultiArray,Float32MultiArray

from cv_bridge import CvBridge
import cv2

from PIL import Image as PILImage
import threading
import numpy as np
import time

from scripts.service_vl import GroundingDINOInfer
import scripts.service_lm as lm
import scripts.ans2json as ans2json
import os
os.environ["QT_QPA_PLATFORM"] = "xcb"
from common_interface.msg import RectDepth,Camera2map
from scripts.memory_builder import MemoryBuilder
import yaml

class ServiceNode(Node):
    def __init__(self):
        super().__init__('detect_vl_node')
        self.bridge = CvBridge()

        self.rgb_image = None
        self.depth_image = None

        self.rect = None
        

        self.obj_list=None
        self.act_list=None

        self.VL = GroundingDINOInfer()
        self.memory_builder = MemoryBuilder()
        self.memory_file = "memory.yaml"
        self.current_room = None
        self.room_pose = [0.0, 0.0, 0.0]  # Default pose, should be updated with actual robot pose
        time.sleep(10)
        self.create_subscription(CompressedImage, '/camera/camera/color/image_raw/compressed', self.rgb_callback, 10)
        self.create_subscription(Image, '/camera/camera/depth/image_rect_raw', self.depth_callback, 10)
        self.create_subscription(Camera2map, '/camera2map', self.camera2map_callback, 10)
        # self.create_subscription()
        self.target_pub = self.create_publisher(RectDepth, 'task/rect_depth', 10)
        self.get_logger().info("ServiceNode node started, waiting for image...")

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

    def camera2map_callback(self, msg):
        """Handle camera to map transformation updates"""
        try:
            # Assuming the message contains [x, y, yaw]
            x, y, yaw = msg.data
            # Update the camera pose in memory builder
            self.memory_builder.update_camera_pose(x, y, yaw)
            self.get_logger().info(f"Updated camera pose: x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}")
        except Exception as e:
            self.get_logger().error(f"Error processing camera pose: {e}")

    def rgb_callback(self, msg):
        try:
            self.rgb_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # Convert to PIL Image for GPT processing
            pil_image = PILImage.fromarray(cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2RGB))
            # Get map building analysis
            map_analysis = lm.gpt_map_build(pil_image)
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
                            dis, wx, wy = self.pix2camera_frame(center)
                            if dis is not None and dis > 0:
                                self.get_logger().info(f"Detected {obj_name} at distance {dis:.2f}m, coordinates ({wx:.2f}, {wy:.2f})")
                                
                                # Add to features list
                                features_with_coords.append({
                                    "object": obj_name,
                                    "Coordinate relative to the world frame": [wx, wy]
                                })
                                
                                # Publish the detection
                                msg = RectDepth()
                                msg.rect = Int32MultiArray()
                                msg.rect.data = rect
                                msg.center = Int32MultiArray()
                                msg.center.data = center
                                msg.frame = time.time()
                                msg.depth = dis
                                msg.coordinate_diff = Float32MultiArray()
                                msg.coordinate_diff.data = [wx, wy]
                                self.target_pub.publish(msg)
                                
                                # Visualize detection
                                cv2.imshow("VL-detect", img_detect)
                                cv2.waitKey(1)
            
                # Save to memory if we have features
                if features_with_coords and "room_type" in map_analysis:
                    self.memory_builder.save_to_memory(map_analysis["room_type"], features_with_coords)
            
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
                    print(f"dis={dis},coordinate=({wx},{wy})")
                    if dis == 0:
                        continue
                    if dis < 0.5:
                        idx += 1
                    else:
                        msg.rect = Int32MultiArray()
                        msg.rect.data = rect

                        msg.center = Int32MultiArray()
                        msg.center.data = center
                        
                        msg.frame = time.time()

                        msg.depth = 3.0 #dis
                        msg.coordinate_diff = Float32MultiArray()


                        msg.coordinate_diff.data = [3.0, 0.7]#[wx, wy]
                        
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
    except KeyboardInterrupt:
        print("⛔ 退出程序")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


