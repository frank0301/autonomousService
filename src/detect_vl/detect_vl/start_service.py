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
from std_msgs.msg import Int32MultiArray,Float32MultiArray,String

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
from common_interface.msg import RectDepth
class ServiceNode(Node):
    def __init__(self):
        super().__init__('detect_vl_node')
        self.bridge = CvBridge()

        self.rgb_image = None
        self.depth_image = None
        self.robot_state = None
        self.rect = None
        
        self.obj_list=None
        self.act_list=None

        self.VL = GroundingDINOInfer()
        
        self.create_subscription(CompressedImage, '/camera/camera/color/image_raw/compressed', self.rgb_callback, 30)
        # self.create_subscription(CompressedImage, '/camera/camera/depth/image_rect_raw/compressed', self.depth_callback, 10)
        self.create_subscription(Image, '/camera/camera/depth/image_rect_raw', self.depth_callback, 30)
        # self.create_subscription()
        self.target_pub = self.create_publisher(RectDepth, 'task/rect_depth', 10)
        self.get_logger().info("ServiceNode node started, waiting for image...")

        self.create_subscription(String, '/robot_state',self.robot_state_update_callback, 10)
        self.showImgTimer=self.create_timer(0.3,self.showImg)
    def showImg(self):
        if self.rgb_image is not None:
            cv2.imshow("rgb img", self.rgb_image)
            cv2.waitKey(1)
    def rgb_callback(self, msg):
        try:
            self.rgb_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"图像处理失败: {e}")

    def depth_callback(self, msg):
        try:
            # if format of 16UC1, do not use 'passthrough'
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            # 假设 depth_image 是 16位 或 32位 float，需要先归一化
            depth_normalized = cv2.normalize(self.depth_image, None, 0, 255, cv2.NORM_MINMAX)
            depth_normalized = np.uint8(depth_normalized)  # 转成 8位

            # 应用伪彩色
            # depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
            # cv2.imshow("depth",self.depth_image)
            # cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"Depth image failed to transfer: {e}")

    def robot_state_update_callback(self, msg):
        self.robot_state = msg.data
        self.get_logger().info(f"{msg.data}")

    def pix2world(self, pix_xy):
            if self.depth_image is None:
                self.get_logger().warn("⚠️ Depth image not yet received.")
                return None,None,None

            center_depth_mm = self.depth_image[pix_xy[1], pix_xy[0]]  # Access depth image using (row, col)
            center_depth_m = center_depth_mm / 1000.0

            # self.get_logger().info(f"📏 Center depth at ({pix_xy[0]},{pix_xy[1]}): {center_depth_m:.3f} m")
            cx,cy = 319.47, 247
            fx,fy = 615.53, 615.53
            pix_x, pix_y = pix_xy 
            wY = (pix_x - cx) * center_depth_m / fx
            wX = center_depth_m #(pix_y - cy) * center_depth_m / fy
            # self.get_logger().info(f"wY:{wY},wX:{wX}")
            return center_depth_m,wX,-wY

    

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
            # question = input("\n enter your question(type Ctrl+C exit):\n> ").strip()
            # if not question:
            #     print("invalid question")
            #     continue
            # answer = ans2json.ans2json(lm.ask_gpt_ll(question))
            # print(f"\n✅ GPT-4o answer: \n{answer}")
            # node.act_list, node.obj_list = answer["actions"], answer["objects"]
            # print(node.act_list, node.obj_list)
            # '''
            #         task here: 
            #         pub the goal
            #             <-
            #         detect the obj VLM <- depth img, rect 
            # '''
            node.obj_list = ['a blue trash']
            node.act_list = ['null']
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
                    dis,wx,wy = node.pix2world(center)
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
                        print("update target!")
                        time.sleep(3)
                elif act and act.lower() != "null":
                    msg.coodinate_diff = [0, 0]
                    msg.theta = float(act)
                    while node.robot_state != "turning":
                        node.target_pub.publish(msg)
                        # rclpy.spin_once(node)
                        time.sleep(0.5)
                    print("waiting turning")
                    time.sleep(5)
                    while node.robot_state != "reachGoal":
                        rclpy.spin_once(node)
                        time.sleep(0.5)
                    idx += 1             
                else:
                    node.get_logger().warn("both null, jump over it!")
                    idx+=1
            print(idx)
            node.get_logger().info("success!")
            # rclpy.spin_once(node)
    except KeyboardInterrupt:
        print("⛔ 退出程序")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


