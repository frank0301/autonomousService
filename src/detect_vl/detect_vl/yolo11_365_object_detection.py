#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import os
import json
from typing import List, Dict, Tuple, Optional
import threading
import time

# ROS message imports
from sensor_msgs.msg import Image, CompressedImage, CameraInfo, PointCloud2
from sensor_msgs_py import point_cloud2
from geometry_msgs.msg import Point, Pose2D
from std_msgs.msg import Header, String, Int32MultiArray, Float32MultiArray
from common_interface.msg import RectDepth, Camera2map

# CV Bridge for image conversion
from cv_bridge import CvBridge

class YOLOv11Object365Detector(Node):
    """
    ROS2 Node for YOLOv11 Object365 object detection.
    Integrates with multiple camera topics for comprehensive object detection.
    """
    
    def __init__(self):
        super().__init__('yolo11_object365_detector')
        
        # Initialize CV Bridge
        self.bridge = CvBridge()
        
        # Initialize model
        self.model = None
        # Get the path to the model file in the package
        package_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(package_dir, "yolo11n_object365.pt")
        self.load_model()
        
        # Image storage
        self.rgb_image = None
        self.depth_image = None
        self.rgb_camera_info = None
        self.depth_camera_info = None
        self.point_cloud = None
        
        # Thread locks for thread safety
        self.rgb_lock = threading.Lock()
        self.depth_lock = threading.Lock()
        self.pc_lock = threading.Lock()
        
        # Detection parameters
        self.confidence_threshold = 0.7
        self.nms_threshold = 0.4
        self.max_detections = 20
        
        # Debug window settings
        self.enable_debug_window = True
        self.debug_window_name = "YOLOv11 Object365 Detection"
        
        # Setup QoS profile for reliable communication
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Initialize subscribers for all camera topics
        self.setup_subscribers(qos_profile)
        
        # Initialize publishers
        self.setup_publishers()
        
        # Detection timer
        self.detection_timer = self.create_timer(0.1, self.process_detections)  # 10 FPS
        
        self.get_logger().info("🚀 YOLOv11 Object365 Detector initialized successfully!")
        self.get_logger().info(f"📊 Model: {self.model_path}")
        self.get_logger().info(f"🎯 Confidence threshold: {self.confidence_threshold}")
        
    def load_model(self):
        """Load the YOLOv11 Object365 model"""
        try:
            self.get_logger().info("🔄 Loading YOLOv11 Object365 model...")
            self.model = YOLO(self.model_path)
            self.get_logger().info("✅ Model loaded successfully!")
            
            # Log model info
            if hasattr(self.model, 'names'):
                num_classes = len(self.model.names)
                self.get_logger().info(f"📋 Model supports {num_classes} Object365 classes")
                
        except Exception as e:
            self.get_logger().error(f"❌ Failed to load model: {e}")
            self.get_logger().error("💡 Make sure you have internet connection and ultralytics installed")
            raise e
    
    def setup_subscribers(self, qos_profile):
        """Setup all camera topic subscribers"""
        
        # RGB Image subscribers
        self.rgb_sub = self.create_subscription(
            Image, 
            '/camera/camera/image_raw', 
            self.rgb_callback, 
            qos_profile
        )
        
        self.rgb_compressed_sub = self.create_subscription(
            CompressedImage, 
            '/camera/camera/image_raw/compressed', 
            self.rgb_compressed_callback, 
            qos_profile
        )
        
        # Depth Image subscribers
        self.depth_sub = self.create_subscription(
            Image, 
            '/camera/camera/depth/image_raw', 
            self.depth_callback, 
            qos_profile
        )
        
        self.depth_compressed_sub = self.create_subscription(
            CompressedImage, 
            '/camera/camera/depth/image_raw/compressed', 
            self.depth_compressed_callback, 
            qos_profile
        )
        
        # Camera Info subscribers
        self.rgb_info_sub = self.create_subscription(
            CameraInfo, 
            '/camera/camera/camera_info', 
            self.rgb_info_callback, 
            qos_profile
        )
        
        self.depth_info_sub = self.create_subscription(
            CameraInfo, 
            '/camera/camera/depth/camera_info', 
            self.depth_info_callback, 
            qos_profile
        )
        
        # Point Cloud subscriber
        self.pointcloud_sub = self.create_subscription(
            PointCloud2, 
            '/camera/camera/points', 
            self.pointcloud_callback, 
            qos_profile
        )
        
        self.get_logger().info("📡 All camera subscribers initialized")
    
    def setup_publishers(self):
        """Setup detection result publishers"""
        
        # Detection results (using custom message format)
        self.detection_pub = self.create_publisher(
            String, 
            '/detection/objects', 
            10
        )
        
        # Custom detection format
        self.rect_depth_pub = self.create_publisher(
            RectDepth, 
            '/detection/rect_depth', 
            10
        )
        
        # Camera to map coordinates
        self.camera2map_pub = self.create_publisher(
            Camera2map, 
            '/detection/camera2map', 
            10
        )
        
        # Detection visualization image
        self.vis_image_pub = self.create_publisher(
            Image, 
            '/detection/visualization', 
            10
        )
        
        # Detection summary
        self.detection_summary_pub = self.create_publisher(
            String, 
            '/detection/summary', 
            10
        )
        
        self.get_logger().info("📤 All publishers initialized")
    
    def rgb_callback(self, msg):
        """Callback for raw RGB images"""
        try:
            with self.rgb_lock:
                self.rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"RGB image conversion failed: {e}")
    
    def rgb_compressed_callback(self, msg):
        """Callback for compressed RGB images"""
        try:
            with self.rgb_lock:
                self.rgb_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Compressed RGB image conversion failed: {e}")
    
    def depth_callback(self, msg):
        """Callback for raw depth images"""
        try:
            with self.depth_lock:
                # Convert depth image and handle 32FC1 format
                depth_cv = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                # Convert to 16-bit depth for better compatibility
                if depth_cv.dtype == np.float32:
                    # Scale float32 depth to 16-bit range
                    depth_cv = (depth_cv * 1000).astype(np.uint16)  # Convert meters to mm
                self.depth_image = depth_cv
        except Exception as e:
            self.get_logger().error(f"Depth image conversion failed: {e}")
    
    def depth_compressed_callback(self, msg):
        """Callback for compressed depth images"""
        try:
            with self.depth_lock:
                # Skip compressed depth for now due to format issues
                # self.depth_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='passthrough')
                pass
        except Exception as e:
            self.get_logger().debug(f"Compressed depth image skipped: {e}")
    
    def rgb_info_callback(self, msg):
        """Callback for RGB camera info"""
        self.rgb_camera_info = msg
    
    def depth_info_callback(self, msg):
        """Callback for depth camera info"""
        self.depth_camera_info = msg
    
    def pointcloud_callback(self, msg):
        """Callback for point cloud data"""
        try:
            with self.pc_lock:
                self.point_cloud = msg
        except Exception as e:
            self.get_logger().error(f"Point cloud processing failed: {e}")
    
    def process_detections(self):
        """Main detection processing function"""
        if self.rgb_image is None or self.model is None:
            return
        
        try:
            # Run YOLOv11 inference
            with self.rgb_lock:
                rgb_copy = self.rgb_image.copy()
            
            results = self.model.predict(
                source=rgb_copy,
                imgsz=640,
                conf=self.confidence_threshold,
                iou=self.nms_threshold,
                max_det=self.max_detections,
                verbose=False,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            
            # Process results
            detections = self.process_yolo_results(results[0], rgb_copy)
            
            # Publish detections
            self.publish_detections(detections, rgb_copy)
            
            # Publish visualization
            self.publish_visualization(detections, rgb_copy)
            
        except Exception as e:
            self.get_logger().error(f"Detection processing failed: {e}")
    
    def process_yolo_results(self, result, image) -> List[Dict]:
        """Process YOLO results and extract detection information"""
        detections = []
        
        if result.boxes is None or len(result.boxes) == 0:
            return detections
        
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()
        
        h, w = image.shape[:2]
        
        for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
            x1, y1, x2, y2 = box.astype(int)
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            
            # Get class name
            class_name = self.model.names[int(cls_id)]
            
            # Calculate depth if available
            depth = self.get_depth_at_point(center_x, center_y)
            
            detection = {
                'bbox': [x1, y1, x2, y2],
                'center': [center_x, center_y],
                'confidence': float(conf),
                'class_id': int(cls_id),
                'class_name': class_name,
                'depth': depth,
                'area': (x2 - x1) * (y2 - y1)
            }
            
            detections.append(detection)
        
        return detections
    
    def get_depth_at_point(self, x: int, y: int) -> Optional[float]:
        """Get depth value at specific pixel coordinates"""
        if self.depth_image is None:
            return None
        
        try:
            with self.depth_lock:
                if (0 <= x < self.depth_image.shape[1] and 
                    0 <= y < self.depth_image.shape[0]):
                    depth_value = self.depth_image[y, x]
                    # Handle different depth formats
                    if depth_value > 0:
                        if self.depth_image.dtype == np.uint16:
                            # Already converted to mm, convert to meters
                            return float(depth_value) / 1000.0
                        elif self.depth_image.dtype == np.float32:
                            # Already in meters
                            return float(depth_value)
                        else:
                            # Default: assume mm
                            return float(depth_value) / 1000.0
        except Exception as e:
            self.get_logger().debug(f"Depth extraction failed: {e}")
        
        return None
    
    def publish_detections(self, detections: List[Dict], image):
        """Publish detection results in multiple formats"""
        if not detections:
            return
        
        # Create detection data for JSON message
        detection_data = {
            'header': {
                'stamp': self.get_clock().now().to_msg().sec + self.get_clock().now().to_msg().nanosec * 1e-9,
                'frame_id': 'camera_link'
            },
            'detections': []
        }
        
        for det in detections:
            detection_info = {
                'bbox': {
                    'x1': int(det['bbox'][0]),
                    'y1': int(det['bbox'][1]), 
                    'x2': int(det['bbox'][2]),
                    'y2': int(det['bbox'][3]),
                    'center_x': int(det['center'][0]),
                    'center_y': int(det['center'][1]),
                    'width': int(det['bbox'][2] - det['bbox'][0]),
                    'height': int(det['bbox'][3] - det['bbox'][1])
                },
                'class_id': int(det['class_id']),
                'class_name': str(det['class_name']),
                'confidence': float(det['confidence']),
                'depth': float(det['depth']) if det['depth'] is not None else None,
                'area': int(det['area'])
            }
            detection_data['detections'].append(detection_info)
        
        # Publish detection data as JSON
        detection_msg = String()
        detection_msg.data = json.dumps(detection_data)
        self.detection_pub.publish(detection_msg)
        
        # Publish custom RectDepth messages for each detection
        for det in detections:
            rect_depth = RectDepth()
            
            # Convert bbox to Int32MultiArray
            rect_depth.rect.data = [int(x) for x in det['bbox']]
            
            # Convert center to Int32MultiArray  
            rect_depth.center.data = [int(x) for x in det['center']]
            
            # Set depth and frame
            rect_depth.depth = float(det['depth']) if det['depth'] is not None else 0.0
            rect_depth.frame = 0.0  # Frame number if needed
            
            # Initialize coordinate_diff as empty for now
            rect_depth.coordinate_diff.data = []
            rect_depth.theta = 0.0
            
            self.rect_depth_pub.publish(rect_depth)
        
        # Publish detection summary
        summary = {
            'timestamp': time.time(),
            'num_detections': len(detections),
            'detections': [
                {
                    'class': str(det['class_name']),
                    'confidence': float(det['confidence']),
                    'center': [int(det['center'][0]), int(det['center'][1])],
                    'depth': float(det['depth']) if det['depth'] is not None else None
                }
                for det in detections
            ]
        }
        
        summary_msg = String()
        summary_msg.data = json.dumps(summary)
        self.detection_summary_pub.publish(summary_msg)
    
    def publish_visualization(self, detections: List[Dict], image):
        """Publish visualization image with bounding boxes and show debug window"""
        vis_image = image.copy()
        
        # Add detection info overlay
        info_text = f"YOLOv11 Object365 | Detections: {len(detections)} | Model: {self.model_path}"
        cv2.putText(vis_image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis_image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            center_x, center_y = det['center']
            class_name = det['class_name']
            confidence = det['confidence']
            depth = det['depth']
            
            # Draw bounding box with different colors based on confidence
            color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255) if confidence > 0.5 else (0, 165, 255)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw center point
            cv2.circle(vis_image, (center_x, center_y), 5, (0, 0, 255), -1)
            
            # Draw label with background
            label = f"{class_name}: {confidence:.2f}"
            if depth is not None:
                label += f" | {depth:.2f}m"
            
            # Calculate text size for background
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Draw background rectangle for text
            cv2.rectangle(vis_image, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
            
            # Draw text
            cv2.putText(
                vis_image, 
                label, 
                (x1, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 0, 0), 
                1
            )
        
        # Show debug window
        if self.enable_debug_window:
            try:
                cv2.imshow(self.debug_window_name, vis_image)
                cv2.waitKey(1)
            except Exception as e:
                self.get_logger().debug(f"Debug window display failed: {e}")
        
        # Convert and publish visualization image
        try:
            vis_msg = self.bridge.cv2_to_imgmsg(vis_image, encoding='bgr8')
            vis_msg.header.stamp = self.get_clock().now().to_msg()
            vis_msg.header.frame_id = 'camera_link'
            self.vis_image_pub.publish(vis_msg)
        except Exception as e:
            self.get_logger().error(f"Visualization publishing failed: {e}")
    
    def cleanup(self):
        """Cleanup function to close debug windows"""
        if self.enable_debug_window:
            cv2.destroyAllWindows()
    
    def get_logger(self):
        return super().get_logger()


def main(args=None):
    """Main function"""
    rclpy.init(args=args)
    
    node = None
    try:
        node = YOLOv11Object365Detector()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("⛔ Shutting down YOLOv11 Object365 Detector...")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if node is not None:
            node.cleanup()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
