#!/usr/bin/env python3

import yaml
import os
import time
import threading
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import math

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

class TopologicalVisualizer:
    def __init__(self, memory_file: Optional[str] = None, update_interval: float = 1.0):
        """
        Initialize the topological graph visualizer
        
        Args:
            memory_file: Path to the memory.yaml file
            update_interval: How often to update the visualization (seconds)
        """
        if memory_file is None:
            # Use workspace root to construct the path
            workspace_root = find_workspace_root()
            self.memory_file = os.path.join(workspace_root, "src", "memory.yaml")
        else:
            self.memory_file = memory_file
            
        self.update_interval = update_interval
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.last_modified_time = 0
        self.animation = None
        
        # Visualization parameters
        self.room_radius = 0.5  # Radius for room circles
        self.feature_radius = 0.25  # Radius for feature circles (half of room radius)
        self.door_radius = 0.15  # Radius for door circles
        
        # Colors
        self.room_color = 'blue'
        self.feature_color = 'red'
        self.door_color = 'yellow'
        self.edge_color = 'black'
        
        # Setup matplotlib
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('X (meters)')
        self.ax.set_ylabel('Y (meters)')
        self.ax.set_title('Topological Graph Visualization')
        
        # Store current visualization elements
        self.room_patches = {}
        self.feature_patches = {}
        self.door_patches = {}
        self.edge_lines = {}
        self.text_annotations = {}
        
        # Animation
        self.ani = None
        
    def load_memory(self) -> Dict[str, Any]:
        """Load memory data from YAML file"""
        if not os.path.exists(self.memory_file):
            return {"nodes": [], "edges": []}
            
        try:
            with open(self.memory_file, 'r') as f:
                data = yaml.safe_load(f)
                return data or {"nodes": [], "edges": []}
        except Exception as e:
            print(f"Error reading memory file: {e}")
            return {"nodes": [], "edges": []}
    
    def clear_visualization(self):
        """Clear all visualization elements"""
        # Clear patches
        for patch in self.room_patches.values():
            if patch in self.ax.patches:
                patch.remove()
        for patch in self.feature_patches.values():
            if patch in self.ax.patches:
                patch.remove()
        for patch in self.door_patches.values():
            if patch in self.ax.patches:
                patch.remove()
        
        # Clear lines
        for line in self.edge_lines.values():
            if line in self.ax.lines:
                line.remove()
        
        # Clear text
        for text in self.text_annotations.values():
            if text in self.ax.texts:
                text.remove()
        
        # Clear collections
        self.room_patches.clear()
        self.feature_patches.clear()
        self.door_patches.clear()
        self.edge_lines.clear()
        self.text_annotations.clear()
    
    def calculate_door_position(self, room1_pos: List[float], room2_pos: List[float]) -> Tuple[float, float]:
        """Calculate door position at the midpoint between two rooms"""
        x1, y1 = room1_pos[0], room1_pos[1]
        x2, y2 = room2_pos[0], room2_pos[1]
        
        # Door at midpoint
        door_x = (x1 + x2) / 2.0
        door_y = (y1 + y2) / 2.0
        
        return door_x, door_y
    
    def update_visualization(self):
        """Update the visualization with current memory data"""
        # Check if file has been modified
        current_mtime = os.path.getmtime(self.memory_file) if os.path.exists(self.memory_file) else 0
        if current_mtime <= self.last_modified_time:
            return
        
        self.last_modified_time = current_mtime
        
        # Load memory data
        memory_data = self.load_memory()
        nodes = memory_data.get('nodes', [])
        edges = memory_data.get('edges', [])
        
        # Clear previous visualization
        self.clear_visualization()
        
        # Create room positions dictionary
        room_positions = {}
        
        # Draw rooms (blue circles)
        for node in nodes:
            room_name = node.get('name', 'unknown')
            room_pose = node.get('pose', [0.0, 0.0, 0.0])
            room_x, room_y = room_pose[0], room_pose[1]
            
            room_positions[room_name] = [room_x, room_y]
            
            # Create room circle
            room_circle = patches.Circle((room_x, room_y), self.room_radius, 
                                       facecolor=self.room_color, alpha=0.7, 
                                       edgecolor='black', linewidth=2)
            self.ax.add_patch(room_circle)
            self.room_patches[room_name] = room_circle
            
            # Add room name label
            room_text = self.ax.text(room_x, room_y, room_name, 
                                   ha='center', va='center', fontsize=10, 
                                   fontweight='bold', color='white')
            self.text_annotations[f"room_{room_name}"] = room_text
            
            # Draw features (red circles) connected to room
            features = node.get('features', [])
            for i, feature in enumerate(features):
                if 'object' in feature and 'Coordinate relative to the world frame' in feature:
                    feature_coords = feature['Coordinate relative to the world frame']
                    if len(feature_coords) >= 2:
                        feature_x, feature_y = feature_coords[0], feature_coords[1]
                        feature_name = feature['object']
                        
                        # Create feature circle
                        feature_circle = patches.Circle((feature_x, feature_y), self.feature_radius,
                                                      facecolor=self.feature_color, alpha=0.8,
                                                      edgecolor='black', linewidth=1)
                        self.ax.add_patch(feature_circle)
                        self.feature_patches[f"{room_name}_{feature_name}_{i}"] = feature_circle
                        
                        # Add feature name label
                        feature_text = self.ax.text(feature_x, feature_y, feature_name,
                                                  ha='center', va='center', fontsize=8,
                                                  color='white', fontweight='bold')
                        self.text_annotations[f"feature_{room_name}_{feature_name}_{i}"] = feature_text
                        
                        # Draw line from room to feature
                        room_to_feature_line, = self.ax.plot([room_x, feature_x], [room_y, feature_y],
                                                           color='gray', alpha=0.5, linewidth=1, linestyle='--')
                        self.edge_lines[f"room_to_feature_{room_name}_{feature_name}_{i}"] = room_to_feature_line
        
        # Draw doors (yellow circles) and room connections
        for edge in edges:
            from_room = edge.get('from')
            to_room = edge.get('to')
            cost = edge.get('cost', 0.0)
            
            if from_room in room_positions and to_room in room_positions:
                room1_pos = room_positions[from_room]
                room2_pos = room_positions[to_room]
                
                # Calculate door position
                door_x, door_y = self.calculate_door_position(room1_pos, room2_pos)
                
                # Create door circle
                door_circle = patches.Circle((door_x, door_y), self.door_radius,
                                           facecolor=self.door_color, alpha=0.9,
                                           edgecolor='black', linewidth=2)
                self.ax.add_patch(door_circle)
                self.door_patches[f"door_{from_room}_{to_room}"] = door_circle
                
                # Add door label
                door_text = self.ax.text(door_x, door_y, "DOOR",
                                       ha='center', va='center', fontsize=8,
                                       fontweight='bold', color='black')
                self.text_annotations[f"door_label_{from_room}_{to_room}"] = door_text
                
                # Draw connection line between rooms through door
                room1_x, room1_y = room1_pos[0], room1_pos[1]
                room2_x, room2_y = room2_pos[0], room2_pos[1]
                
                # Line from room1 to door
                line1, = self.ax.plot([room1_x, door_x], [room1_y, door_y],
                                    color=self.edge_color, linewidth=2, alpha=0.8)
                self.edge_lines[f"edge1_{from_room}_{to_room}"] = line1
                
                # Line from door to room2
                line2, = self.ax.plot([door_x, room2_x], [door_y, room2_y],
                                    color=self.edge_color, linewidth=2, alpha=0.8)
                self.edge_lines[f"edge2_{from_room}_{to_room}"] = line2
                
                # Add cost label at door position
                cost_text = self.ax.text(door_x, door_y + self.door_radius + 0.1, f"cost: {cost}",
                                        ha='center', va='bottom', fontsize=7,
                                        color='black', alpha=0.7)
                self.text_annotations[f"cost_{from_room}_{to_room}"] = cost_text
        
        # Update plot limits to show all elements
        if room_positions:
            all_x = [pos[0] for pos in room_positions.values()]
            all_y = [pos[1] for pos in room_positions.values()]
            
            # Add some padding
            x_min, x_max = min(all_x) - 2, max(all_x) + 2
            y_min, y_max = min(all_y) - 2, max(all_y) + 2
            
            self.ax.set_xlim(x_min, x_max)
            self.ax.set_ylim(y_min, y_max)
        
        # Update display
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        print(f"Updated visualization: {len(nodes)} rooms, {len(edges)} connections")
    
    def animate(self, frame):
        """Animation function for real-time updates"""
        self.update_visualization()
        return []
    
    def start_visualization(self):
        """Start the real-time visualization"""
        print(f"Starting topological graph visualization...")
        print(f"Monitoring file: {self.memory_file}")
        print(f"Update interval: {self.update_interval} seconds")
        print("Press Ctrl+C to stop")
        
        try:
            # Create animation
            self.ani = FuncAnimation(self.fig, self.animate, 
                                   interval=self.update_interval * 1000,  # Convert to milliseconds
                                   blit=False, repeat=True)
            
            # Show the plot
            plt.show()
            
        except KeyboardInterrupt:
            print("\nVisualization stopped by user")
        except Exception as e:
            print(f"Error in visualization: {e}")
        finally:
            plt.close()
    
    def save_snapshot(self, filename: str = "topological_graph.png"):
        """Save current visualization as an image"""
        self.update_visualization()
        self.fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Snapshot saved as {filename}")


def main():
    """Main function to run the visualizer"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Topological Graph Visualizer')
    parser.add_argument('--memory-file', default=None, 
                       help='Path to memory.yaml file (default: auto-detect from workspace root)')
    parser.add_argument('--update-interval', type=float, default=1.0,
                       help='Update interval in seconds (default: 1.0)')
    parser.add_argument('--save-snapshot', action='store_true',
                       help='Save a snapshot and exit instead of running real-time')
    parser.add_argument('--snapshot-filename', default='topological_graph.png',
                       help='Filename for snapshot (default: topological_graph.png)')
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = TopologicalVisualizer(
        memory_file=args.memory_file,
        update_interval=args.update_interval
    )
    
    if args.save_snapshot:
        # Save snapshot and exit
        visualizer.save_snapshot(args.snapshot_filename)
    else:
        # Start real-time visualization
        visualizer.start_visualization()


if __name__ == '__main__':
    main() 