#!/usr/bin/env python3

"""
Example usage of the TopologicalVisualizer

This script demonstrates how to use the topological visualizer and shows
the expected structure of the memory.yaml file.
"""

import yaml
import os
from topological_visualizer import TopologicalVisualizer

def create_example_memory_file():
    """Create an example memory.yaml file for testing"""
    
    # Example memory data structure
    example_memory = {
        "nodes": [
            {
                "name": "kitchen",
                "pose": [0.0, 0.0, 0.0],
                "features": [
                    {
                        "object": "refrigerator",
                        "Coordinate relative to the world frame": [1.5, 0.5]
                    },
                    {
                        "object": "sink",
                        "Coordinate relative to the world frame": [-1.0, 1.0]
                    },
                    {
                        "object": "table",
                        "Coordinate relative to the world frame": [0.5, -1.2]
                    }
                ]
            },
            {
                "name": "living_room",
                "pose": [5.0, 0.0, 0.0],
                "features": [
                    {
                        "object": "sofa",
                        "Coordinate relative to the world frame": [4.5, 1.0]
                    },
                    {
                        "object": "tv",
                        "Coordinate relative to the world frame": [6.0, -0.5]
                    }
                ]
            },
            {
                "name": "bedroom",
                "pose": [0.0, 5.0, 0.0],
                "features": [
                    {
                        "object": "bed",
                        "Coordinate relative to the world frame": [0.5, 4.5]
                    },
                    {
                        "object": "desk",
                        "Coordinate relative to the world frame": [-1.0, 6.0]
                    }
                ]
            }
        ],
        "edges": [
            {
                "from": "kitchen",
                "to": "living_room",
                "cost": 5.0
            },
            {
                "from": "kitchen",
                "to": "bedroom",
                "cost": 5.0
            },
            {
                "from": "living_room",
                "to": "bedroom",
                "cost": 7.0
            }
        ]
    }
    
    # Save to file
    with open("example_memory.yaml", 'w') as f:
        yaml.dump(example_memory, f, default_flow_style=False)
    
    print("Created example_memory.yaml with sample data")
    return "example_memory.yaml"

def demonstrate_visualizer():
    """Demonstrate the visualizer with example data"""
    
    # Create example memory file
    memory_file = create_example_memory_file()
    
    print("\n=== Topological Graph Visualizer Demo ===")
    print("This will show:")
    print("- Blue circles: Rooms (kitchen, living_room, bedroom)")
    print("- Red circles: Features within rooms (furniture, objects)")
    print("- Yellow circles: Doors connecting rooms")
    print("- Gray dashed lines: Connections from rooms to their features")
    print("- Black solid lines: Room connections through doors")
    print("\nPress Ctrl+C to stop the visualization")
    
    # Create and start visualizer
    visualizer = TopologicalVisualizer(
        memory_file=memory_file,
        update_interval=2.0  # Update every 2 seconds
    )
    
    try:
        visualizer.start_visualization()
    except KeyboardInterrupt:
        print("\nDemo stopped by user")
    
    # Clean up
    if os.path.exists(memory_file):
        os.remove(memory_file)
        print(f"Cleaned up {memory_file}")

def show_usage_instructions():
    """Show usage instructions for the visualizer"""
    
    print("\n=== Usage Instructions ===")
    print("\n1. Real-time visualization:")
    print("   python topological_visualizer.py")
    print("   python topological_visualizer.py --memory-file your_memory.yaml")
    print("   python topological_visualizer.py --update-interval 0.5")
    
    print("\n2. Save snapshot:")
    print("   python topological_visualizer.py --save-snapshot")
    print("   python topological_visualizer.py --save-snapshot --snapshot-filename my_graph.png")
    
    print("\n3. Programmatic usage:")
    print("   from topological_visualizer import TopologicalVisualizer")
    print("   visualizer = TopologicalVisualizer('src/memory.yaml')")
    print("   visualizer.start_visualization()")
    
    print("\n4. Expected memory.yaml structure:")
    print("   nodes:")
    print("     - name: room_name")
    print("       pose: [x, y, yaw]")
    print("       features:")
    print("         - object: feature_name")
    print("           Coordinate relative to the world frame: [x, y]")
    print("   edges:")
    print("     - from: room1_name")
    print("       to: room2_name")
    print("       cost: distance_value")

if __name__ == '__main__':
    print("Topological Graph Visualizer - Example Usage")
    print("=" * 50)
    
    # Show usage instructions
    show_usage_instructions()
    
    # Ask user if they want to see the demo
    try:
        response = input("\nWould you like to see a demo? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            demonstrate_visualizer()
        else:
            print("Demo skipped. You can run the visualizer directly with:")
            print("python topological_visualizer.py")
    except KeyboardInterrupt:
        print("\nExiting...") 