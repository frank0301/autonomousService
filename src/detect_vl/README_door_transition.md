# Door Transition Detection for Room Classification

This document describes the implementation of door transition detection to prevent automatic room type classification unless the robot has passed through a door.

## Overview

The system now includes intelligent room transition detection that prevents the robot from classifying new room types unless it has physically passed through a door. This ensures more accurate semantic mapping by avoiding false room classifications when the robot is simply moving within the same room.

## Key Features

### 1. Door Detection
- **Automatic Detection**: Uses the VLM (Vision Language Model) to detect doors in camera images
- **Distance Tracking**: Monitors the robot's distance to detected doors
- **Proximity Sensing**: Triggers when the robot gets within 2 meters of a door

### 2. Room Transition Logic
- **Transition Detection**: Recognizes when the robot has passed through a door
- **State Management**: Tracks door detection and room transition states
- **Classification Control**: Prevents new room classification without door transitions

### 3. Configuration Parameters
```python
self.door_detection_threshold = 3.0  # seconds to consider door detection valid
self.door_detection_distance_threshold = 2.0  # meters - distance to consider door "passed through"
```

## How It Works

### Door Detection Process
1. **Continuous Monitoring**: The system runs door detection every 0.5 seconds
2. **VLM Analysis**: Uses Grounding DINO to detect "door" objects in camera images
3. **Distance Calculation**: Computes the robot's distance to detected doors using depth information
4. **State Tracking**: Maintains door detection and transition states

### Room Classification Rules
1. **First Room**: Always allows the first room classification
2. **Same Room**: Allows classification if the proposed room type matches the current room
3. **New Room**: Only allows new room classification if a door transition has been detected
4. **Feature Storage**: Still saves detected features to the current room even if new classification is blocked

### Transition Detection Logic
```
Door Detected (within 2m) → Door Proximity State
↓
Robot Moves Away (>2m) → Room Transition Detected
↓
New Room Classification Allowed
```

## Usage

### Automatic Operation
The system works automatically once the `ServiceNode` is running. It will:
- Continuously detect doors in the environment
- Track door proximity and transitions
- Control room classification based on door transitions
- Log all detection and classification events

### Manual Override
For testing or manual control, you can reset the room transition state:

#### Using ROS Service
```bash
# Call the reset service
rosservice call /reset_room_transition
```

#### Using the Test Script
```bash
# Run the test script
python3 test_door_transition.py

# Available commands:
# reset - Manually reset room transition state
# status - Check current status
# exit - Exit the test
```

## Logging and Debugging

The system provides comprehensive logging:

- 🚪 Door detection events
- ✅ Room transition confirmations
- 🚫 Blocked room classifications
- 📝 Feature storage operations
- 🔄 Manual state resets

### Example Log Output
```
[INFO] [detect_vl_node]: 🚪 Door detected at distance 1.85m
[INFO] [detect_vl_node]: 🚪 Door proximity detected - preparing for room transition
[INFO] [detect_vl_node]: ✅ Room transition detected - door passed through
[INFO] [detect_vl_node]: ✅ Room transition confirmed - classifying new room: kitchen
[INFO] [detect_vl_node]: ✅ Room 'kitchen' classified and saved to memory
```

## Configuration

You can adjust the detection parameters in the `ServiceNode.__init__()` method:

```python
# Door detection parameters
self.door_detection_threshold = 3.0  # seconds
self.door_detection_distance_threshold = 2.0  # meters

# Detection frequency
self.door_detection_timer = self.create_timer(0.5, self.detect_doors)  # 0.5 seconds
```

## Integration with Existing System

The door transition detection integrates seamlessly with the existing system:

1. **Memory Builder**: Features are still saved to the appropriate room
2. **VLM Detection**: Uses the same VLM system for door detection
3. **ROS Integration**: Maintains all existing ROS topics and services
4. **Backward Compatibility**: Doesn't break existing functionality

## Testing

To test the door transition detection:

1. Start the main service node
2. Run the test script: `python3 test_door_transition.py`
3. Use the reset command to manually trigger room transitions
4. Observe the logging output for detection events

## Troubleshooting

### Common Issues

1. **No Door Detection**: Ensure the VLM model is properly loaded and camera images are available
2. **False Transitions**: Adjust the distance threshold if transitions are triggered too easily
3. **Missed Transitions**: Decrease the distance threshold or increase detection frequency

### Debug Information

Use the `get_room_transition_status()` method to get current state information:

```python
status = node.get_room_transition_status()
print(f"Current room: {status['current_room']}")
print(f"Door detected: {status['door_detected']}")
print(f"Transition detected: {status['room_transition_detected']}")
``` 