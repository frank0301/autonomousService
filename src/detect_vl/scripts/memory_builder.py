import yaml
import os
import math
import numpy as np
from typing import List, Dict, Any, Tuple

class MemoryBuilder:
    def __init__(self, memory_file: str = "memory.yaml"):
        self.memory_file = memory_file
        self.memory_data = {
            "nodes": [],
            "edges": []
        }
        self._load_memory()
        self.camera_pose = None  # [x, y, yaw]

    def _load_memory(self):
        """Load existing memory if it exists"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r') as f:
                    self.memory_data = yaml.safe_load(f) or {"nodes": [], "edges": []}
            except yaml.YAMLError as e:
                print(f"Error reading memory file: {e}")
                self.memory_data = {"nodes": [], "edges": []}

    def _calculate_distance(self, pose1: List[float], pose2: List[float]) -> float:
        """Calculate Euclidean distance between two poses"""
        return math.sqrt((pose1[0] - pose2[0])**2 + (pose1[1] - pose2[1])**2)

    def _transform_to_map_frame(self, camera_coords: List[float]) -> List[float]:
        """Transform coordinates from camera frame to map frame using the camera pose"""
        if self.camera_pose is None:
            return camera_coords
        
        cam_x, cam_y, cam_yaw = self.camera_pose
        diff_x, diff_y = camera_coords[0], camera_coords[1]
        
        # Apply the transformation equations
        target_x = cam_x + diff_x * math.cos(cam_yaw) - diff_y * math.sin(cam_yaw)
        target_y = cam_y + diff_x * math.sin(cam_yaw) + diff_y * math.cos(cam_yaw)
        
        return [target_x, target_y, camera_coords[2]]  # Keep original z coordinate

    def _is_feature_unique(self, new_feature: Dict[str, Any], existing_features: List[Dict[str, Any]]) -> bool:
        """Check if a feature is unique based on its type and name"""
        for existing_feature in existing_features:
            if (existing_feature.get('type') == new_feature.get('type') and 
                existing_feature.get('name') == new_feature.get('name')):
                return False
        return True

    def _filter_features_by_distance(self, features: List[Dict[str, Any]], room_pose: List[float]) -> List[Dict[str, Any]]:
        """Filter features that are within 4 meters of the room pose"""
        filtered_features = []
        for feature in features:
            if 'Coordinate relative to the world frame' in feature:
                feature_coords = feature['Coordinate relative to the world frame']
                distance = self._calculate_distance(room_pose[:2], feature_coords[:2])  # Only use x,y coordinates
                if distance <= 4.0:  # 4 meters threshold
                    filtered_features.append(feature)
        return filtered_features

    def _update_edges(self):
        """Update all edges based on current room poses"""
        self.memory_data["edges"] = []
        nodes = self.memory_data["nodes"]
        
        # Create edges between all rooms
        for i, node1 in enumerate(nodes):
            for node2 in nodes[i+1:]:
                distance = self._calculate_distance(node1["pose"], node2["pose"])
                # Only create edge if rooms are within reasonable distance (e.g., 5 meters)
                if distance < 1.0:
                    edge = {
                        "from": node1["name"],
                        "to": node2["name"],
                        "cost": round(distance, 2)
                    }
                    self.memory_data["edges"].append(edge)

    def update_camera_pose(self, x: float, y: float, yaw: float):
        """Update the camera pose in the map frame"""
        self.camera_pose = [x, y, yaw]

    def save_to_memory(self, room_type: str, features_with_coords: List[Dict[str, Any]], room_pose: List[float] = [0.0, 0.0, 0.0]):
        """Save or update room features in memory"""
        # Transform room pose to map frame
        map_room_pose = self._transform_to_map_frame(room_pose)
        
        # Transform feature coordinates to map frame
        transformed_features = []
        for feature in features_with_coords:
            if 'Coordinate relative to the world frame' in feature:
                camera_coords = feature['Coordinate relative to the world frame']
                map_coords = self._transform_to_map_frame(camera_coords)
                feature['Coordinate relative to the world frame'] = map_coords
                transformed_features.append(feature)

        # Filter features by distance
        filtered_features = self._filter_features_by_distance(transformed_features, map_room_pose)
        
        # Check if room already exists
        room_exists = False
        for node in self.memory_data["nodes"]:
            if node["name"] == room_type:
                # Update existing room with unique features
                existing_features = node["features"]
                for feature in filtered_features:
                    if self._is_feature_unique(feature, existing_features):
                        existing_features.append(feature)
                node["features"] = existing_features
                node["pose"] = map_room_pose
                room_exists = True
                break

        if not room_exists:
            # Create new room node with filtered features
            new_room = {
                "name": room_type,
                "pose": map_room_pose,
                "features": filtered_features
            }
            self.memory_data["nodes"].append(new_room)

        # Update edges after room update
        self._update_edges()

        # Save to file
        try:
            with open(self.memory_file, 'w') as f:
                yaml.dump(self.memory_data, f, default_flow_style=False)
            print(f"Updated memory file with {room_type} features and edges")
        except Exception as e:
            print(f"Error saving to memory file: {e}")

    def get_room_features(self, room_type: str) -> List[Dict[str, Any]]:
        """Get features for a specific room"""
        for node in self.memory_data["nodes"]:
            if node["name"] == room_type:
                return node["features"]
        return []

    def get_all_rooms(self) -> List[str]:
        """Get list of all room names"""
        return [node["name"] for node in self.memory_data["nodes"]]

    def get_connected_rooms(self, room_type: str) -> List[Tuple[str, float]]:
        """Get list of rooms connected to the specified room and their costs"""
        connected = []
        for edge in self.memory_data["edges"]:
            if edge["from"] == room_type:
                connected.append((edge["to"], edge["cost"]))
            elif edge["to"] == room_type:
                connected.append((edge["from"], edge["cost"]))
        return connected

    def update_room_pose(self, room_type: str, new_pose: List[float]):
        """Update the pose of a specific room"""
        map_pose = self._transform_to_map_frame(new_pose)
        for node in self.memory_data["nodes"]:
            if node["name"] == room_type:
                node["pose"] = map_pose
                self._update_edges()
                self.save_to_memory(room_type, node["features"], map_pose)
                break 