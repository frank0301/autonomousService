#!/usr/bin/env python3
"""
Test script for MemoryBuilder to demonstrate loading and saving memory.yaml from src folder
"""

import sys
import os

# Add the scripts directory to the path so we can import memory_builder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from memory_builder import MemoryBuilder

def test_memory_builder():
    """Test the MemoryBuilder functionality"""
    
    print("=== Testing MemoryBuilder with src/memory.yaml ===")
    
    # Create MemoryBuilder instance (will automatically use src/memory.yaml)
    memory_builder = MemoryBuilder()
    
    # Print the memory file path
    print(f"Memory file path: {memory_builder.get_memory_file_path()}")
    
    # Check if file exists
    if os.path.exists(memory_builder.get_memory_file_path()):
        print("✅ Memory file exists")
        
        # Load and display current memory
        print("\n=== Current Memory Content ===")
        rooms = memory_builder.get_all_rooms()
        print(f"Found {len(rooms)} rooms: {rooms}")
        
        for room in rooms:
            features = memory_builder.get_room_features(room)
            print(f"\nRoom: {room}")
            print(f"  Features: {len(features)}")
            for feature in features:
                obj_name = feature.get('object', 'unknown')
                if 'Coordinate relative to the world frame' in feature:
                    coords = feature['Coordinate relative to the world frame']
                    print(f"    - {obj_name}: ({coords[0]:.2f}, {coords[1]:.2f})")
                else:
                    print(f"    - {obj_name}: no coordinates")
            
            # Show connected rooms
            connected = memory_builder.get_connected_rooms(room)
            if connected:
                print(f"  Connected to: {connected}")
    else:
        print("❌ Memory file does not exist")
    
    # Test backup functionality
    print("\n=== Testing Backup Functionality ===")
    memory_builder.backup_memory()
    
    # Test reload functionality
    print("\n=== Testing Reload Functionality ===")
    memory_builder.reload_memory()
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_memory_builder() 