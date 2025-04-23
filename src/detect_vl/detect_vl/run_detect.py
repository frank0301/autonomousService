import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
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
openai.api_key = os.getenv("OPENAI_API_KEY")

SYSTEM_PROMPT_WORD='''
I am about to give a command to the robot. Please help me identify the mentioned object in the instruction, 
and output the pixel coordinates of its bounding box: the upper-left corner and the lower-right corner. 
the image sized [640, 480]
Return the result in JSON format only.

Example:
If the command is: "Go to the trash can and turn right."
You should output:
{
    "object": "trash can",
    "xy": [[475, 298], [501, 322]]
}

Now, here is the new instruction:
'''

from common_interface.msg import RectDepth
class DetectVLNode(Node):
    
    def __init__(self):
        super().__init__('detect_vl_node')
        self.bridge = CvBridge()

        self.rgb_image = None
        self.start_point = None
        self.end_point = None

        self.create_subscription(Image, 'camera/rgb/image_raw', self.rgb_callback, 10)
        self.create_subscription(Image, 'camera/depth/image_raw', self.depth_callback, 10)

        self.create_publisher(RectDepth, 'task/rect_depth', 10)
        self.get_logger().info("✅ DetectVL node started, waiting for image...")

    def rgb_callback(self, msg):
        try:
            self.rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # show img
            cv2.rectangle(self.rgb_image, self.start_point, self.end_point, (255, 0, 0), 2)
            cv2.imshow("RGB Camera Feed", self.rgb_image)
            cv2.waitKey(1)  # update the window
        except Exception as e:
            self.get_logger().error(f"RGB failed to transfer image: {e}")
    def depth_callback(self, msg):
        try:
            # if format of 16UC1, do not use 'passthrough'
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f"Depth image failed to transfer: {e}")

    def ask_gpt4o_with_image(self, cv2_img, question):
        pil_img = PILImage.fromarray(cv2_img)
        buffered = io.BytesIO()
        pil_img.save(buffered, format="JPEG")
        base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
        image = 'data:image/jpeg;base64,' + base64_image
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

        (x1, y1), (x2, y2) = xy
        x1 = x1
        x2 = x2
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(self.depth_image.shape[1]-1, x2), min(self.depth_image.shape[0]-1, y2)
        self.start_point,self.end_point = [x1,y1],[x2,y2]
        # extract center
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        center_depth = self.depth_image[cy, cx]

        # extranc average
        region = self.depth_image[y1:y2, x1:x2]
        valid_region = region[region > 0]  # filter invalid
        mean_depth = np.mean(valid_region) if valid_region.size > 0 else 0

        self.get_logger().info(f"📏 Center depth at ({cx},{cy}): {center_depth} mm")
        self.get_logger().info(f"📊 Mean depth in region: {mean_depth:.2f} mm")

        return {
            "center": (cx, cy),
            "center_depth": int(center_depth),
            "mean_depth": float(mean_depth)
        }
def main(args=None):
    rclpy.init(args=args)
    node = DetectVLNode()

    # ROS2 spin
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    # main question loops
    try:
        while True:
            question = input("\n enter your question(type Ctrl+C exit):\n> ").strip()
            if not question:
                print("invalid question")
                continue

            if node.rgb_image is None:
                print("no frames input, waiting for stream...")
                continue
            
            print("📤 sending to GPT-4o, waiting for response...")
            try:
                answer = node.ask_gpt4o_with_image(node.rgb_image, question)
#                 answer = '''
# ```json
# {
#     "object": "keyboard",
#     "xy": [[450, 350], [580, 400]]
# }
# ```
# '''
                print(f"\n✅ GPT-4o answer: {answer}")
                import re
                print(node.get_depth_from_box(json.loads(re.sub(r"```", "", re.sub(r"```json", "", answer).strip()).strip()).get("xy",None)))

            except Exception as e:
                print(f"❌ failed GPT-4o : {e}")

    except KeyboardInterrupt:
        print("\n🛑 exit.")
    finally:
        node.destroy_node()
        rclpy.shutdown()
