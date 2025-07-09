# Autonomous Service - Memory Map System

## 🎯 Purpose

This branch implements an intelligent **semantic memory mapping system** for autonomous robots. The system combines computer vision, large language models (LLMs), and visual language models (VLMs) to:

- **Detect and classify objects** in the robot's environment
- **Build semantic memory maps** of rooms and spaces
- **Track robot pose** and transform coordinates between camera and map frames
- **Enable room transition detection** through door identification
- **Create topological maps** with connected rooms and features

## 🏗️ Architecture

### Core Components

1. **Vision-Language Model (VLM) Integration**
   - Uses Grounding DINO for object detection
   - Processes natural language queries for object identification

2. **Large Language Model (LLM) Integration**
   - GPT-4 Vision for semantic understanding
   - Room classification and feature extraction
   - Natural language task interpretation

3. **Memory Builder System**
   - Coordinate transformation (camera frame ↔ map frame)
   - Semantic memory storage in YAML format
   - Room connectivity and topological mapping

4. **ROS 2 Integration**
   - Camera pose tracking via `/camera2map` topic
   - Image processing from RGB and depth cameras
   - Real-time object detection and navigation

## 🚀 Features

### ✅ Implemented Features

- **Real-time Object Detection**: VLM-based detection with depth information
- **Semantic Room Classification**: Automatic room type identification
- **Coordinate Transformation**: Camera frame to map frame conversion
- **Memory Persistence**: YAML-based semantic memory storage
- **Room Transition Detection**: Door-based room change detection
- **Topic Monitoring**: Automatic warning system for missing camera pose data
- **Error Handling**: Robust validation for images and messages

### 🔧 Key Capabilities

- **Multi-modal Processing**: Combines RGB, depth, and pose data
- **Natural Language Interface**: Accepts queries in plain English
- **Automatic Memory Building**: Continuously updates semantic maps
- **Real-time Monitoring**: Tracks system health and data flow
- **Coordinate Accuracy**: Precise transformation using robot yaw

## 📁 File Structure

```
src/
├── detect_vl/
│   ├── detect_vl/
│   │   ├── start_service.py      # Main service node
│   │   └── run_detect.py         # YOLO-based detection node
│   └── scripts/
│       ├── memory_builder.py     # Memory management system
│       ├── service_vl.py         # VLM integration
│       ├── service_lm.py         # LLM integration
│       └── ans2json.py           # Response parsing
├── common_interface/
│   └── msg/
│       ├── Camera2map.msg        # Camera pose message
│       └── RectDepth.msg         # Detection result message
└── goundingDinoTest.py           # VLM testing script
```

## 🛠️ Installation & Setup

### Prerequisites

```bash
# ROS 2 (tested with Humble)
sudo apt install ros-humble-desktop

# Python dependencies
pip install torch torchvision
pip install transformers
pip install opencv-python
pip install openai
pip install ultralytics
pip install pyyaml
pip install pillow
```

### Building the Workspace

```bash
# Clone and build
cd ~/autonomousService
colcon build

# Source the workspace
source install/setup.bash
```

### Environment Variables

```bash
# Set OpenAI API key for LLM features
export OPENAI_API_KEY="your_openai_api_key_here"
```

## 🎮 Usage

### 1. Starting the Main Service

```bash
# Source the workspace
source ~/autonomousService/install/setup.bash

# Run the main service node
python3 src/detect_vl/detect_vl/start_service.py
```

### 2. Running YOLO Detection

```bash
# Alternative detection using YOLO
python3 src/detect_vl/detect_vl/run_detect.py
```

### 3. Testing VLM Detection

```bash
# Test Grounding DINO object detection
python3 src/goundingDinoTest.py
```

## 📊 System Monitoring

### Topic Monitoring

The system automatically monitors the `/camera2map` topic and warns if:
- No messages received for 5+ seconds
- Camera pose data is missing
- Coordinate transformation fails

### Status Checking

```python
# Check camera2map topic status
node.check_camera2map_status()

# Check room transition status
node.get_room_transition_status()
```

## 🔧 Configuration

### Message Types

**Camera2map Message:**
```yaml
coordinate:
  data: [x, y, yaw]  # Robot pose in map frame
```

**RectDepth Message:**
```yaml
rect: [x1, y1, x2, y2]      # Bounding box
center: [cx, cy]           # Object center
depth: float               # Distance to object
coordinate_diff: [wx, wy]  # Camera frame coordinates
```

### Memory File Format

```yaml
nodes:
  - name: "kitchen"
    pose: [1.2, 3.4, 0.5]  # [x, y, yaw]
    features:
      - object: "refrigerator"
        Coordinate relative to the world frame: [1.1, 3.3]
      - object: "sink"
        Coordinate relative to the world frame: [1.3, 3.5]
edges:
  - from: "kitchen"
    to: "living_room"
    cost: 2.5
```

## 🐛 Troubleshooting

### Common Issues

1. **"Camera2map object has no attribute 'theta'"**
   - ✅ **Fixed**: Use `msg.coordinate.data[2]` for yaw

2. **"OpenCV imshow error: size.width>0 && size.height>0"**
   - ✅ **Fixed**: Added image validation before display

3. **"No /camera2map messages received"**
   - Check if camera pose publisher is running
   - Verify topic is being published: `ros2 topic list | grep camera2map`

4. **Message type 'common_interface/msg/Camera2map' is invalid**
   - Source the workspace: `source install/setup.bash`
   - Rebuild if needed: `colcon build`

### Debug Commands

```bash
# Check topic info
ros2 topic info /camera2map

# Echo camera pose data
ros2 topic echo /camera2map

# List available interfaces
ros2 interface list | grep common_interface

# Check node status
ros2 node list
ros2 node info /detect_vl_node
```

## 🔄 Workflow

1. **Start the system** with camera and pose data
2. **System automatically** detects objects and builds memory
3. **Room transitions** are detected through door identification
4. **Semantic maps** are continuously updated and saved
5. **Coordinate transformations** enable accurate object localization

## 🎯 Use Cases

- **Autonomous Navigation**: Use semantic memory for path planning
- **Object Search**: Find specific objects in mapped environments
- **Room Classification**: Automatically identify and categorize spaces
- **Memory Persistence**: Maintain knowledge across robot sessions
- **Multi-room Exploration**: Build connected topological maps

## 🤝 Contributing

When contributing to this branch:

1. **Test coordinate transformations** thoroughly
2. **Validate message parsing** with actual ROS topics
3. **Update memory file format** documentation if changed
4. **Add error handling** for new features
5. **Test with real camera data** when possible

## 📝 Notes

- **Coordinate System**: Uses standard ROS coordinate frames
- **Memory Persistence**: Saves to `memory.yaml` in workspace root
- **Real-time Processing**: Designed for live camera feeds
- **Error Recovery**: Graceful handling of missing data
- **Scalability**: Supports multiple rooms and objects

---

**Branch**: `memory_map`  
**Last Updated**: July 2024  
**Status**: ✅ Production Ready
