# YOLOv11 Object365 Integration for ROS2

This package provides a comprehensive ROS2 integration for YOLOv11 object detection using the Object365 dataset. The detector can process multiple camera feeds simultaneously and provides rich detection outputs including depth information.

## Features

- **YOLOv11 Object365 Model**: Uses the pretrained model from Hugging Face
- **Multi-Camera Support**: Processes RGB, depth, and point cloud data
- **Real-time Detection**: High-performance inference with configurable parameters
- **Depth Fusion**: Integrates depth information with 2D detections
- **Multiple Output Formats**: Standard ROS2 messages and custom formats
- **Visualization**: Real-time detection visualization
- **Thread-Safe**: Handles multiple camera streams safely

## Model Information

- **Model**: YOLOv11n Object365
- **Source**: https://huggingface.co/NRtred/yolo11n_object365
- **Classes**: 365 object classes from Object365 dataset
- **Input Size**: 640x640 pixels
- **Framework**: Ultralytics YOLO

## Installation

### Prerequisites

1. ROS2 (tested with Humble)
2. Python 3.8+
3. CUDA (optional, for GPU acceleration)

### Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install ROS2 dependencies (if not already installed)
sudo apt install ros-humble-vision-msgs ros-humble-cv-bridge ros-humble-sensor-msgs-py
```

### Build the Package

```bash
cd /path/to/your/workspace
colcon build --packages-select detect_vl
source install/setup.bash
```

## Usage

### Basic Usage

```bash
# Run the detector node
ros2 run detect_vl yolo11_object365_detector
```

### Using Launch File

```bash
# Launch with default parameters
ros2 launch detect_vl yolo11_object365.launch.py

# Launch with custom parameters
ros2 launch detect_vl yolo11_object365.launch.py \
    confidence_threshold:=0.5 \
    nms_threshold:=0.3 \
    max_detections:=50 \
    device:=cuda
```

### Configuration

The detector can be configured using the YAML configuration file:

```bash
# Edit configuration
nano config/yolo11_object365.yaml
```

## Camera Topics

The detector subscribes to the following camera topics:

### Input Topics
- `/camera/camera/image_raw` - Raw RGB images
- `/camera/camera/image_raw/compressed` - Compressed RGB images
- `/camera/camera/depth/image_raw` - Raw depth images
- `/camera/camera/depth/image_raw/compressed` - Compressed depth images
- `/camera/camera/camera_info` - RGB camera calibration info
- `/camera/camera/depth/camera_info` - Depth camera calibration info
- `/camera/camera/points` - Point cloud data

### Output Topics
- `/detection/objects` - Detection2DArray messages
- `/detection/rect_depth` - Custom RectDepth messages
- `/detection/camera2map` - Camera to map coordinate transforms
- `/detection/visualization` - Annotated detection images
- `/detection/summary` - JSON summary of detections

## Message Types

### Detection2DArray
Standard ROS2 vision message containing:
- Bounding boxes
- Class IDs and confidence scores
- Object hypotheses

### RectDepth (Custom)
Custom message containing:
- Bounding box coordinates
- Center point coordinates
- Depth value
- Frame information

### Detection Summary (JSON)
```json
{
  "timestamp": 1234567890.123,
  "num_detections": 5,
  "detections": [
    {
      "class": "person",
      "confidence": 0.85,
      "center": [320, 240],
      "depth": 2.5
    }
  ]
}
```

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `confidence_threshold` | 0.3 | Minimum confidence for detections |
| `nms_threshold` | 0.4 | Non-Maximum Suppression threshold |
| `max_detections` | 100 | Maximum detections per frame |
| `device` | auto | Inference device (auto/cpu/cuda) |
| `detection_frequency` | 10.0 | Detection processing frequency (Hz) |

## Performance

### Hardware Requirements
- **Minimum**: CPU-only inference
- **Recommended**: NVIDIA GPU with CUDA support
- **Memory**: 4GB+ RAM for model loading

### Performance Metrics
- **CPU (Intel i7)**: ~5-10 FPS
- **GPU (RTX 3080)**: ~30-60 FPS
- **Model Size**: ~6MB (YOLOv11n)

## Troubleshooting

### Common Issues

1. **Model Loading Failed**
   ```bash
   # Check internet connection
   ping huggingface.co
   
   # Verify ultralytics installation
   python -c "from ultralytics import YOLO; print('OK')"
   ```

2. **CUDA Not Available**
   ```bash
   # Check CUDA installation
   nvidia-smi
   
   # Install PyTorch with CUDA
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Camera Topics Not Found**
   ```bash
   # List available topics
   ros2 topic list | grep camera
   
   # Check if camera driver is running
   ros2 node list | grep camera
   ```

### Debug Mode

Enable debug logging:
```bash
ros2 run detect_vl yolo11_object365_detector --ros-args --log-level debug
```

## Integration Examples

### With Navigation Stack
```python
# Subscribe to detections
detection_sub = self.create_subscription(
    Detection2DArray,
    '/detection/objects',
    self.detection_callback,
    10
)
```

### With Custom Processing
```python
# Process detection summary
def detection_callback(self, msg):
    summary = json.loads(msg.data)
    for detection in summary['detections']:
        if detection['class'] == 'person':
            # Process person detection
            self.handle_person_detection(detection)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv11
- [Object365 Dataset](https://www.objects365.org/) for the training data
- [Hugging Face](https://huggingface.co/NRtred/yolo11n_object365) for the model hosting
