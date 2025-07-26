# Large Language Vision Model Navigation in a Dynamic Enviornment
# Autonomous Service - Intelligent Semantic Memory Mapping System

## Abstract

This thesis presents an advanced **Intelligent Semantic Memory Mapping System** for autonomous mobile robots, integrating state-of-the-art computer vision, large language models (LLMs), and visual language models (VLMs). The system enables robots to build comprehensive semantic understanding of their environment through real-time object detection, room classification, and topological mapping, with enhanced accuracy through user-provided environmental context.

## 🎯 Research Objectives

This system addresses the critical challenge of enabling autonomous robots to:
- **Build semantic understanding** of complex indoor environments
- **Maintain persistent memory** across navigation sessions
- **Adapt to diverse environments** through context-aware analysis
- **Enable natural language interaction** for task specification
- **Provide accurate coordinate transformations** between sensor and world frames

## 🏗️ System Architecture

### Core Components

#### 1. **Vision-Language Model (VLM) Integration**
- **Model**: Grounding DINO (IDEA-Research/grounding-dino-tiny)
- **Function**: Natural language-driven object detection
- **Performance**: Real-time inference with GPU acceleration
- **Features**: Bounding box detection, confidence scoring, multi-object queries

#### 2. **Large Language Model (LLM) Integration**
- **Model**: GPT-4o-mini-2024-07-18
- **Function**: Semantic understanding and room classification
- **Features**: Context-aware analysis, environment-specific object prioritization
- **Input**: Multi-modal (text + image) processing

#### 3. **Memory Builder System**
- **Coordinate Transformation**: Camera frame ↔ Map frame conversion
- **Semantic Storage**: YAML-based persistent memory
- **Topological Mapping**: Room connectivity and feature relationships
- **Transition Detection**: Door-based room change identification

#### 4. **ROS 2 Integration**
- **Topic Management**: Camera pose tracking via `/camera2map`
- **Image Processing**: RGB and depth camera integration
- **Real-time Communication**: Publisher/subscriber architecture
- **Service Interface**: Manual control and reset capabilities

## 🚀 Key Features

### ✅ **Context-Aware Semantic Mapping**
- **User Context Integration**: Environment-specific guidance (warehouse, supermarket, hospital, etc.)
- **Adaptive Object Detection**: Prioritizes environment-relevant features
- **Dynamic Room Classification**: Context-informed space categorization

### ✅ **Real-time Performance Optimization**
- **Display Lag Reduction**: Intelligent frame rate control (100ms intervals)
- **CPU Usage Optimization**: Efficient OpenCV waitKey timing (30ms)
- **Memory Management**: Optimized image processing pipeline

### ✅ **Robust Error Handling**
- **Topic Monitoring**: Automatic detection of missing camera pose data
- **Graceful Degradation**: System continues operation with partial data
- **Comprehensive Logging**: Detailed status tracking and debugging

### ✅ **Advanced Coordinate Systems**
- **Multi-frame Support**: Camera, map, and world coordinate transformations
- **Pose Integration**: Real-time robot position and orientation tracking
- **Depth Integration**: Accurate 3D object localization

## 📁 Project Structure

```
src/
├── detect_vl/
│   ├── detect_vl/
│   │   ├── start_service.py      # Main service node with context integration
│   │   └── run_detect.py         # YOLO-based alternative detection
│   └── scripts/
│       ├── memory_builder.py     # Memory management and coordinate transformation
│       ├── service_vl.py         # VLM integration (Grounding DINO)
│       ├── service_lm.py         # LLM integration with context-aware prompts
│       └── ans2json.py           # Response parsing and validation
├── common_interface/
│   └── msg/
│       ├── Camera2map.msg        # Camera pose message definition
│       └── RectDepth.msg         # Detection result message definition
└── goundingDinoTest.py           # VLM testing and validation script
```

## 🛠️ Technical Implementation

### Prerequisites

```bash
# ROS 2 Humble Installation
sudo apt install ros-humble-desktop

# Python Dependencies
pip install torch torchvision
pip install transformers
pip install opencv-python
pip install openai
pip install ultralytics
pip install pyyaml
pip install pillow
```

### Environment Configuration

```bash
# OpenAI API Configuration
export OPENAI_API_KEY="your_openai_api_key_here"

# Workspace Setup
cd ~/autonomousService
colcon build
source install/setup.bash
```

## 🎮 System Operation

### 1. **Main Service Execution**

```bash
# Launch the primary semantic mapping service
python3 src/detect_vl/detect_vl/start_service.py
```

**Interactive Workflow:**
1. **Environment Context Input**: User specifies environment type (e.g., "warehouse", "supermarket")
2. **Task Specification**: Natural language navigation commands
3. **Automatic Processing**: VLM detection + LLM analysis + Memory building
4. **Real-time Feedback**: Visual detection display and coordinate output

### 2. **Alternative Detection Methods**

```bash
# YOLO-based object detection
python3 src/detect_vl/detect_vl/run_detect.py

# VLM testing and validation
python3 src/goundingDinoTest.py
```

## 📊 System Monitoring & Diagnostics

### **Topic Health Monitoring**
- **Camera Pose Tracking**: Automatic detection of `/camera2map` topic inactivity
- **Warning System**: 5-second threshold for missing pose data
- **Recovery Detection**: Automatic notification of topic restoration

### **Status Query Interface**
```python
# Camera pose status
node.check_camera2map_status()

# Room transition monitoring
node.get_room_transition_status()

# Memory builder diagnostics
node.memory_builder.get_door_detection_status()
```

## 🔧 Configuration & Message Types

### **Camera2map Message Structure**
```yaml
coordinate:
  data: [x, y, yaw]  # Robot pose in map frame (meters, meters, radians)
```

### **RectDepth Message Structure**
```yaml
rect: [x1, y1, x2, y2]      # Object bounding box (pixels)
center: [cx, cy]           # Object center coordinates (pixels)
depth: float               # Distance to object (meters)
coordinate_diff: [wx, wy]  # Camera frame coordinates (meters)
frame: float              # Timestamp
```

### **Semantic Memory Format**
```yaml
nodes:
  - name: "warehouse_storage"
    pose: [1.2, 3.4, 0.5]  # [x, y, yaw] in map frame
    features:
      - object: "storage_rack"
        Coordinate relative to the world frame: [1.1, 3.3]
      - object: "forklift"
        Coordinate relative to the world frame: [1.3, 3.5]
edges:
  - from: "warehouse_storage"
    to: "loading_area"
    cost: 2.5
```

## 🔬 Research Contributions

### **1. Context-Aware Semantic Mapping**
- **Novel Approach**: Integration of user-provided environmental context
- **Improved Accuracy**: Environment-specific object prioritization
- **Adaptive Classification**: Dynamic room type identification

### **2. Performance Optimization**
- **Display Efficiency**: Intelligent frame rate control for VLM detection
- **Resource Management**: Optimized CPU usage and memory allocation
- **Real-time Processing**: Sub-second response times for object detection

### **3. Robust System Design**
- **Fault Tolerance**: Graceful handling of missing sensor data
- **Topic Monitoring**: Automatic detection of system health issues
- **Error Recovery**: Self-healing mechanisms for data loss

### **4. Multi-modal Integration**
- **VLM + LLM Synergy**: Combined visual and linguistic understanding
- **Coordinate Transformation**: Accurate multi-frame coordinate systems
- **Memory Persistence**: Long-term semantic knowledge retention

## 🐛 Troubleshooting & Debugging

### **Common Issues & Solutions**

1. **Camera Pose Data Missing**
   ```bash
   # Verify topic publication
   ros2 topic list | grep camera2map
   ros2 topic echo /camera2map
   ```

2. **VLM Detection Performance**
   - **Symptom**: Display lag or high CPU usage
   - **Solution**: Automatic optimization implemented (100ms intervals, 30ms waitKey)

3. **Memory File Access**
   ```bash
   # Check memory file location
   ls -la src/memory.yaml
   # Verify YAML syntax
   python3 -c "import yaml; yaml.safe_load(open('src/memory.yaml'))"
   ```

### **Debug Commands**
```bash
# System diagnostics
ros2 node list
ros2 node info /detect_vl_node
ros2 topic info /camera2map

# Memory inspection
cat src/memory.yaml
```

## 🔄 System Workflow

### **1. Initialization Phase**
- Camera and depth sensor calibration
- VLM model loading (Grounding DINO)
- Memory system initialization
- Topic subscription establishment

### **2. Context Acquisition**
- User environment specification
- Context-aware prompt generation
- Environment-specific object prioritization

### **3. Continuous Operation**
- Real-time image processing
- VLM object detection
- LLM semantic analysis
- Memory building and persistence
- Coordinate transformation

### **4. Room Transition Detection**
- Door identification via VLM
- Transition state management
- New room classification
- Topological map updates

## 🎯 Applications & Use Cases

### **Industrial Applications**
- **Warehouse Automation**: Inventory tracking and navigation
- **Hospital Robotics**: Medical equipment localization
- **Retail Automation**: Product search and shelf management

### **Research Applications**
- **Semantic SLAM**: Environment understanding and mapping
- **Human-Robot Interaction**: Natural language task specification
- **Multi-modal AI**: Integration of vision and language models

### **Educational Applications**
- **Robotics Education**: Hands-on AI and robotics learning
- **Computer Vision Research**: Object detection and classification
- **Natural Language Processing**: Multi-modal AI systems

## 📈 Performance Metrics

### **Detection Performance**
- **VLM Accuracy**: Grounding DINO object detection precision
- **Processing Speed**: Real-time inference capabilities
- **Memory Efficiency**: Optimized storage and retrieval

### **System Reliability**
- **Uptime**: Continuous operation monitoring
- **Error Recovery**: Automatic fault detection and resolution
- **Data Integrity**: Robust memory persistence

## 🤝 Future Work

### **Planned Enhancements**
1. **Multi-robot Coordination**: Distributed semantic mapping
2. **Dynamic Environment Adaptation**: Real-time context learning
3. **Advanced Topological Mapping**: Hierarchical space representation
4. **Human-in-the-loop Learning**: Interactive map refinement

### **Research Extensions**
1. **3D Semantic Mapping**: Volumetric environment representation
2. **Temporal Memory**: Time-aware feature tracking
3. **Cross-modal Learning**: Enhanced vision-language integration

## 📝 Thesis Information

**Research Area**: Autonomous Robotics and Artificial Intelligence  
**Focus**: Semantic Memory Mapping and Multi-modal AI Integration  
**Technology Stack**: ROS 2, Python, PyTorch, OpenAI GPT, Computer Vision  
**Status**: ✅ Production Ready - Thesis Implementation Complete

---

**Author**: Tamir Basson  
**Institution**: Ben Gurion University 
**Department**: Mechanical Engineering  
**Supervisor**: Prof. Amir Shapiro 
**Date**: [Current Date]  
**Version**: 2.0 - Context-Aware Semantic Mapping System
