import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import cv2
import base64
import io
from PIL import Image as PILImage
import openai
import os
import numpy as np
import yaml
from ultralytics import YOLO
from common_interface.msg import RectDepth

def find_workspace_root():
    """Find the workspace root directory by looking for common workspace markers"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Look for workspace root by going up directories
    while current_dir != os.path.dirname(current_dir):  # Stop at filesystem root
        # Check if this looks like a workspace root
        if (os.path.exists(os.path.join(current_dir, 'src')) and 
            os.path.exists(os.path.join(current_dir, 'README.md'))):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    
    # If we can't find workspace root, use current working directory
    return os.getcwd()

openai.api_key = os.getenv("OPENAI_API_KEY")

SYSTEM_PROMPT_WORD = '''
You are an advanced multimodal assistant integrated into a mobile robot. 
Your task is to identify static semantic landmarks in the image for navigation.
Use the bounding boxes and class names shown in the image.
Only use the center coordinates shown as text on the image.
Output this JSON format:
{
  "find_in_img": [
    {"object": "semantic class name", "center": [x, y]},
    ...
  ],
  "target": {
    "object": "target object name",
    "center": [x, y]
  }
}
'''

SYSTEM_PROMPT_MAP = '''
You are a robot assistant tasked with building a semantic memory map.
Based on the objects and their locations in the image, create a YAML-style
topological memory graph. Each location should include:
- name of the semantic area (if deducible)
- list of identified static objects
- their pixel coordinates (center)
Output format:
nodes:
  - name: "location_name"
    features:
      - object: "object_name"
        center: [x, y]
'''



class DetectVLNode(Node):
    def __init__(self):
        super().__init__('detect_vl_node')
        self.bridge = CvBridge()
        self.rgb_image = None
        self.depth_image = None
        self.model = YOLO('yolov8n.pt')
        self.create_subscription(CompressedImage, '/camera/camera/color/image_raw/compressed', self.rgb_callback, 10)
        self.create_subscription(Image, '/camera/camera/depth/image_rect_raw', self.depth_callback, 10)
        self.create_publisher(RectDepth, 'task/rect_depth', 10)
        self.get_logger().info("✅ DetectVL node started.")

    def rgb_callback(self, msg):
        try:
            self.rgb_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            results = self.model.predict(source=self.rgb_image, imgsz=640, conf=0.3, verbose=False, device='cuda')
            result = results[0]
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            for box, conf, cls in zip(boxes, confs, classes):
                x1, y1, x2, y2 = map(int, box)
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                label = f"{self.model.names[int(cls)]} ({center_x},{center_y})"
                cv2.rectangle(self.rgb_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(self.rgb_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow("YOLO Detection Feed", self.rgb_image)
            cv2.waitKey(1)

            # Ask GPT for semantic object summary
            landmarks = self.ask_gpt4o_with_image(self.rgb_image, "Identify landmarks")
            self.get_logger().info(f"🔍 GPT Landmark Output: {landmarks}")

            # Ask GPT for semantic map generation
            semantic_map = self.ask_gpt4o_for_map(self.rgb_image, "Create semantic memory map")
            self.save_yaml_memory(semantic_map)
            self.get_logger().info("📂 Semantic map saved to src/memory.yaml")

        except Exception as e:
            self.get_logger().error(f"Image processing failed: {e}")

    def depth_callback(self, msg):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            cv2.imshow("depth", self.depth_image)
        except Exception as e:
            self.get_logger().error(f"Depth image failed to transfer: {e}")

    def ask_gpt4o_with_image(self, cv2_img, question):
        return self._ask_gpt4_vision(cv2_img, question, SYSTEM_PROMPT_WORD)

    def ask_gpt4o_for_map(self, cv2_img, question):
        return self._ask_gpt4_vision(cv2_img, question, SYSTEM_PROMPT_MAP)

    def _ask_gpt4_vision(self, cv2_img, question, system_prompt):
        pil_img = PILImage.fromarray(cv2_img)
        buffered = io.BytesIO()
        pil_img.save(buffered, format="JPEG")
        base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ]
        response = openai.ChatCompletion.create(
            model="gpt-4-vision-preview",
            messages=messages,
            max_tokens=700
        )
        return response.choices[0].message.content

    def save_yaml_memory(self, yaml_text):
        try:
            data = yaml.safe_load(yaml_text)
            # Use workspace root to construct the path
            workspace_root = find_workspace_root()
            memory_file_path = os.path.join(workspace_root, "src", "memory.yaml")
            with open(memory_file_path, "w") as f:
                yaml.dump(data, f)
        except yaml.YAMLError as exc:
            self.get_logger().error(f"YAML parsing error: {exc}")

def main(args=None):
    rclpy.init(args=args)
    node = DetectVLNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("⛔ Exit the program")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()