import yaml
import os
import math
from typing import List, Dict, Any, Tuple

class MemoryBuilder:
    def __init__(self, memory_file: str = "memory.yaml"):
        self.memory_file = memory_file
        self.memory_data = {
            "nodes": [],
            "edges": []
        }
        self._load_memory()

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

    def save_to_memory(self, room_type: str, features_with_coords: List[Dict[str, Any]], room_pose: List[float] = [0.0, 0.0, 0.0]):
        """Save or update room features in memory"""
        # Create new room node
        new_room = {
            "name": room_type,
            "pose": room_pose,
            "features": features_with_coords
        }

        # Check if room already exists
        room_exists = False
        for node in self.memory_data["nodes"]:
            if node["name"] == room_type:
                # Update existing room
                node["features"] = features_with_coords
                node["pose"] = room_pose
                room_exists = True
                break

        if not room_exists:
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
        for node in self.memory_data["nodes"]:
            if node["name"] == room_type:
                node["pose"] = new_pose
                self._update_edges()
                self.save_to_memory(room_type, node["features"], new_pose)
                break 