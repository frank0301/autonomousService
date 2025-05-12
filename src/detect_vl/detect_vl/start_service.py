# File: detect_vl_node.py
import argparse
import threading
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Int32MultiArray, Float32MultiArray, String
from cv_bridge import CvBridge
import cv2
import numpy as np

from scripts.service_vl import GroundingDINOInfer
import scripts.service_lm as lm
import scripts.ans2json as ans2json
from common_interface.msg import RectDepth

class ServiceNode(Node):
    def __init__(self, use_gui=False):
        super().__init__('detect_vl_node')
        self.bridge = CvBridge()
        self.rgb_image = None
        self.depth_image = None
        self.rect = None
        self.act_list = []
        self.obj_list = []
        self.VL = GroundingDINOInfer()

        # Camera topics
        self.create_subscription(
            CompressedImage,
            '/camera/camera/color/image_raw/compressed',
            self.rgb_callback,
            30
        )
        self.create_subscription(
            Image,
            '/camera/camera/depth/image_rect_raw',
            self.depth_callback,
            30
        )

        # GUI question subscription
        if use_gui:
            self.create_subscription(
                String,
                'service_question',
                self.question_callback,
                10
            )

        self.target_pub = self.create_publisher(RectDepth, 'task/rect_depth', 10)
        self.get_logger().info("ServiceNode started, waiting for input...")

    def rgb_callback(self, msg):
        try:
            self.rgb_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"RGB callback error: {e}")

    def depth_callback(self, msg):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f"Depth callback error: {e}")

    def pix2world(self, pix_xy):
        if self.depth_image is None:
            self.get_logger().warn("Depth image not yet received.")
            return None, None, None
        center_depth_mm = self.depth_image[pix_xy[1], pix_xy[0]]
        center_depth_m = center_depth_mm / 1000.0
        self.get_logger().info(f"Depth at {pix_xy}: {center_depth_m:.3f} m")
        cx, cy = 320, 240
        fx, fy = 385, 385
        pix_x, pix_y = pix_xy
        wY = (pix_x - cx) * center_depth_m / fx
        wX = center_depth_m
        return center_depth_m, wX, wY

    def question_callback(self, msg: String):
        self.process_question(msg.data)

    def process_question(self, question: str):
        question = question.strip()
        if not question:
            self.get_logger().warn("Received empty question.")
            return
        self.get_logger().info(f"Processing question: {question}")
        answer = ans2json.ans2json(lm.ask_gpt_ll(question))
        self.act_list, self.obj_list = answer.get("actions", []), answer.get("objects", [])

        idx = 0
        while idx < len(self.act_list):
            obj = self.obj_list[idx]
            act = self.act_list[idx]
            msg = RectDepth()
            if obj and obj.lower() != "null":
                img_detect, rect, center = self.VL.infer(self.rgb_image, obj + ".")
                if rect is None:
                    continue
                cv2.imshow("VL-detect", img_detect)
                cv2.waitKey(1)
                dis, wx, wy = self.pix2world(center)
                if dis is None:
                    continue
                if dis < 0.1:
                    idx += 1
                else:
                    msg.rect = Int32MultiArray(data=rect)
                    msg.center = Int32MultiArray(data=center)
                    msg.frame = time.time()
                    msg.depth = dis
                    msg.coordinate_diff = Float32MultiArray(data=[wx, wy])
                    self.target_pub.publish(msg)
            elif act and act.lower() != "null":
                msg.coordinate_diff = Float32MultiArray(data=[0.0, 0.0])
                msg.theta = float(act)
                self.target_pub.publish(msg)
                idx += 1
                time.sleep(5)
            else:
                self.get_logger().warn("both null, jump over it!")
                idx += 1
        self.get_logger().info("Question processing complete.")

    def run_cli(self):
        try:
            while True:
                q = input("Enter your question (Ctrl+C to exit):\n> ").strip()
                self.process_question(q)
        except KeyboardInterrupt:
            self.get_logger().info("Exiting CLI mode.")


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-gui', action='store_true', help='Use GUI service_question subscription')
    
    # ✅ Accept unknown ROS 2 launch args
    parsed, _ = parser.parse_known_args()

    rclpy.init(args=args)
    node = ServiceNode(use_gui=parsed.use_gui)

    if parsed.use_gui:
        rclpy.spin(node)
    else:
        import threading
        thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
        thread.start()
        node.run_cli()

    node.destroy_node()
    rclpy.shutdown()