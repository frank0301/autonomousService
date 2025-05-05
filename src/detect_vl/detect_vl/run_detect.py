import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2
import base64
import io
from PIL import Image as PILImage
import openai
import os
import threading
import time
import numpy as np

import json
from ultralytics import YOLO
openai.api_key = os.getenv("OPENAI_API_KEY")

SYSTEM_PROMPT_WORD='''
You are an advanced multimodal assistant integrated into a mobile robot. 
I'm about to give a cmd to the robot to reach a pleace.
Your task is to identify static semantic landmarks in the given image for navigation purposes.
Bounding boxes and class names are provided by an object detector and drawn on the image.
The center coordinates are already displayed on the image next to each bounding box as text (e.g., (140, 230)).
Do not calculate the center coordinates yourself. Directly use the text from the image (via OCR or visual reading).
Only include static, non-movable objects (e.g., door, table, chair, shelf, cabinet, wall painting).
Exclude movable items like people, pets, cups, bottles, bags, laptops, or similar.
Please output the result using the following JSON structure:
{
    "find_in_img": [
        {
            "object": "semantic class name",
            "center": [center_x, center_y]  
        },
        ...
    ],
    "target": {
        "object": "target object name",
        "center": [center_x, center_y]  
    }
}
'''
import os
os.environ["QT_QPA_PLATFORM"] = "xcb"
from common_interface.msg import RectDepth
class DetectVLNode(Node):
    def __init__(self):
        super().__init__('detect_vl_node')
        self.bridge = CvBridge()

        self.rgb_image = None
        self.depth_image = None
        self.model = YOLO('yolov8n.pt')#YOLO('src/detect_vl/detect_vl/yolov11n.pt')
        self.start_point = None
        self.end_point = None

        self.create_subscription(CompressedImage, '/camera/camera/color/image_raw/compressed', self.rgb_callback, 10)
        self.create_subscription(Image, '/camera/camera/depth/image_rect_raw', self.depth_callback, 10)
        # self.create_subscription(CompressedImage, '/camera/camera/depth/image_rect_raw/compressed', self.yolo_callback, 10)
        self.create_publisher(RectDepth, 'task/rect_depth', 10)
        self.get_logger().info("✅ DetectVL node started, waiting for image...")

    def rgb_callback(self, msg):
        try:
            self.rgb_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            results = self.model.predict(source=self.rgb_image, imgsz=640, conf=0.3, verbose=False, device='cuda')
            result = results[0]
            boxes = result.boxes.xyxy.cpu().numpy()  # 转为numpy数组
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            
            for box, conf, cls in zip(boxes, confs, classes):
                x1, y1, x2, y2 = map(int, box)
                label = f"{self.model.names[int(cls)]} {conf:.2f}"
                cv2.rectangle(self.rgb_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(self.rgb_image, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow("YOLO Detection Feed", self.rgb_image)
            cv2.waitKey(1)           
        except Exception as e:
            self.get_logger().error(f"图像处理失败: {e}")

    def depth_callback(self, msg):
        try:
            # if format of 16UC1, do not use 'passthrough'
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            cv2.imshow("depth",self.depth_image)
        except Exception as e:
            self.get_logger().error(f"Depth image failed to transfer: {e}")

    def yolo_callback(self, msg):
        try:
            self.detect_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # show img
            # cv2.rectangle(self.rgb_image, self.start_point, self.end_point, (255, 0, 0), 2)
            cv2.imshow("detect_image Feed", self.detect_image)
            cv2.waitKey(1)  # update the window
        except Exception as e:
            self.get_logger().error(f"RGB failed to transfer image: {e}")
            
    def ask_gpt4o_with_image(self, cv2_img, question):
        cv2.imwrite("detect_img.jpg",cv2_img)
        pil_img = PILImage.fromarray(cv2_img)
        buffered = io.BytesIO()
        pil_img.save(buffered, format="JPEG")
        base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        input = None
        response = openai.responses.create(
            model="gpt-4o-mini-2024-07-18",
            input=[
                {
                    "role": "user",
                    "content": [
                        { "type": "input_text", "text": SYSTEM_PROMPT_WORD + question},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    ],
                }
            ],
        )
        print(input,'\n')
        return response.output_text

    def get_depth_from_box(self, xy):
            if self.depth_image is None:
                self.get_logger().warn("⚠️ Depth image not yet received.")
                return None

            center_depth_mm = self.depth_image[xy[1], xy[0]]  # Access depth image using (row, col)
            center_depth_m = center_depth_mm / 1000.0

            self.get_logger().info(f"📏 Center depth at ({xy[0]},{xy[1]}): {center_depth_m:.3f} m")
            return center_depth_m

def main(args=None):
    rclpy.init(args=args)
    node = DetectVLNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("⛔ 退出程序")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


# def main(args=None):
#     rclpy.init(args=args)
#     node = DetectVLNode()

#     # ROS2 spin
#     threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
#     try:
#         while True:
#             # continue
#             if node.rgb_image is None:
#                 continue
#     #         results = node.model.predict(source=node.rgb_image, imgsz=640, conf=0.3, verbose=False, device='cuda')

#     #         # ✅ 取第一个result（因为predict返回的是list）
#     #         result = results[0]
#     #         boxes = result.boxes.xyxy.cpu().numpy()  # 转为numpy数组
#     #         confs = result.boxes.conf.cpu().numpy()
#     #         classes = result.boxes.cls.cpu().numpy()

#     #         # ✅ 在图像上画框
#     #         for box, conf, cls in zip(boxes, confs, classes):
#     #             x1, y1, x2, y2 = map(int, box)
#     #             label = f"{node.model.names[int(cls)]} {conf:.2f}"
#     #             cv2.rectangle(node.rgb_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#     #             cv2.putText(node.rgb_image, label, (x1, y1 - 10),
#     #                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

#     #         # ✅ 显示推理后的图像
#             cv2.imshow("YOLO Detection Feed", node.rgb_image)
#             cv2.waitKey(10)
#     # main question loops
#     # try:
#     #     while True:
#     #         question = input("\n enter your question(type Ctrl+C exit):\n> ").strip()
#     #         if not question:
#     #             print("invalid question")
#     #             continue

#     #         if node.detect_image is None:
#     #             print("no frames input, waiting for stream...")
#     #             continue
            
#     #         print("📤 sending to GPT-4o, waiting for response...")
#     #         try:
#     #             answer = node.ask_gpt4o_with_image(node.detect_image, question)
#     #             print(f"\n✅ GPT-4o answer: {answer}")
#     #             import re
#     #             clean_str = re.sub(r"```", "", re.sub(r"```json", "", answer).strip()).strip()
#     #             xy = None
#     #             # json.loads(re.sub(r"```", "", re.sub(r"```json", "", answer).strip()).strip()).get("xy",None)
#     #             # json.loads(re.sub(r"```", "", re.sub(r"```json", "", answer).strip(
#     #             # 解析 JSON
#     #             data = json.loads(clean_str)

#     #             # 分别提取 object 和 center
#     #             objects = [item['object'] for item in data['find_in_img']]
#     #             centers = [item['center'] for item in data['find_in_img']]

#     #             # 目标 target
#     #             target_object = data['target']['object']
#     #             target_centsite-packages/detect_vl/run_detect.py", line 197, in main
#     # rclpy.shutdown()
#     except Exception as e:
#         print(f"❌ failed GPT-4o : {e}")

#     # except KeyboardInterrupt:site-packages/detect_vl/run_detect.py", line 197, in main
#         rclpy.shutdown()
#         print("\n🛑 exit.")
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()
