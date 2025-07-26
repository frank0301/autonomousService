#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import os
import yaml
import time

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA

class SemanticMapRvizPublisher(Node):
    def __init__(self):
        super().__init__('semantic_map_rviz_publisher')

        # Path to memory.yaml (adjust if needed)
        self.memory_file = os.path.expanduser('src/memory.yaml')

        # Publisher for MarkerArray
        self.marker_pub = self.create_publisher(MarkerArray, 'semantic_map_markers', 10)

        # Keep track of last modification time to avoid redundant updates
        self.last_mtime = 0.0

        # Counter for generating unique marker IDs
        self.id_counter = 0

        # Timer to periodically check for updates (every 1 second)
        self.create_timer(1.0, self.timer_callback)

    def timer_callback(self):
        """Check if memory.yaml was modified, then load and publish markers."""
        if not os.path.exists(self.memory_file):
            self.get_logger().warning('Memory file not found: %s' % self.memory_file)
            return

        current_mtime = os.path.getmtime(self.memory_file)
        if current_mtime <= self.last_mtime:
            return  # No changes since last publish

        self.last_mtime = current_mtime
        memory_data = self.load_memory()
        markers = self.create_markers(memory_data)
        self.marker_pub.publish(markers)
        self.get_logger().info('Published {} markers'.format(len(markers.markers)))

    def load_memory(self):
        """Load semantic map data from memory.yaml."""
        try:
            with open(self.memory_file, 'r') as f:
                data = yaml.safe_load(f)
                return data or {"nodes": [], "edges": []}
        except Exception as e:
            self.get_logger().error('Failed to read memory file: {}'.format(e))
            return {"nodes": [], "edges": []}

    def next_id(self):
        """Return a unique integer ID for each marker."""
        marker_id = self.id_counter
        self.id_counter += 1
        return marker_id

    def create_markers(self, memory_data):
        """
        Convert rooms (nodes) and edges from memory.yaml into RViz markers.
        - Rooms are blue semi-transparent spheres with text labels.
        - Features are red semi-transparent spheres with text labels.
        - Edges are black lines with cost labels.
        """
        marker_array = MarkerArray()
        self.id_counter = 0  # Reset ID counter each time

        nodes = memory_data.get('nodes', [])
        edges = memory_data.get('edges', [])

        # Build a map from room name to its XY pose for drawing edges
        room_positions = {node['name']: (node['pose'][0], node['pose'][1]) for node in nodes}

        # 1) Create room markers and their text labels
        for room in nodes:
            room_name = room['name']
            x, y, _ = room['pose']

            # Sphere marker for room
            room_marker = Marker()
            room_marker.header.frame_id = 'map'
            room_marker.header.stamp = self.get_clock().now().to_msg()
            room_marker.ns = 'rooms'
            room_marker.id = self.next_id()
            room_marker.type = Marker.SPHERE
            room_marker.action = Marker.ADD
            room_marker.pose.position.x = x
            room_marker.pose.position.y = y
            room_marker.pose.position.z = 0.0
            room_marker.pose.orientation.w = 1.0
            room_marker.scale.x = 0.6  # diameter in meters
            room_marker.scale.y = 0.6
            room_marker.scale.z = 0.01  # very thin for 2D map
            room_marker.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.5)  # blue, semi-transparent
            marker_array.markers.append(room_marker)

            # Text marker for room name
            text_marker = Marker()
            text_marker.header.frame_id = 'map'
            text_marker.header.stamp = self.get_clock().now().to_msg()
            text_marker.ns = 'room_labels'
            text_marker.id = self.next_id()
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position.x = x
            text_marker.pose.position.y = y
            text_marker.pose.position.z = 0.2  # slightly above sphere
            text_marker.pose.orientation.w = 1.0
            text_marker.scale.z = 0.2  # text height
            text_marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)  # white text
            text_marker.text = room_name
            marker_array.markers.append(text_marker)

            # 2) Create feature markers and their labels
            for feature in room.get('features', []):
                # Use field names exactly as in memory.yaml
                coords = feature.get('Coordinate relative to the world frame', None)
                if coords is None:
                    continue

                fx, fy = coords[0], coords[1]
                feature_name = feature.get('object', 'Unknown')

                # Sphere marker for feature
                feat_marker = Marker()
                feat_marker.header.frame_id = 'map'
                feat_marker.header.stamp = self.get_clock().now().to_msg()
                feat_marker.ns = 'features'
                feat_marker.id = self.next_id()
                feat_marker.type = Marker.SPHERE
                feat_marker.action = Marker.ADD
                feat_marker.pose.position.x = fx
                feat_marker.pose.position.y = fy
                feat_marker.pose.position.z = 0.0
                feat_marker.pose.orientation.w = 1.0
                feat_marker.scale.x = 0.2  # smaller diameter for feature
                feat_marker.scale.y = 0.2
                feat_marker.scale.z = 0.01
                feat_marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.5)  # red, semi-transparent
                marker_array.markers.append(feat_marker)

                # Text marker for feature name
                feat_text = Marker()
                feat_text.header.frame_id = 'map'
                feat_text.header.stamp = self.get_clock().now().to_msg()
                feat_text.ns = 'feature_labels'
                feat_text.id = self.next_id()
                feat_text.type = Marker.TEXT_VIEW_FACING
                feat_text.action = Marker.ADD
                feat_text.pose.position.x = fx
                feat_text.pose.position.y = fy
                feat_text.pose.position.z = 0.15  # just above the small sphere
                feat_text.pose.orientation.w = 1.0
                feat_text.scale.z = 0.1  # smaller text
                feat_text.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
                feat_text.text = feature_name
                marker_array.markers.append(feat_text)

        # 3) Create edge markers between connected rooms
        for edge in edges:
            from_name = edge.get('from')
            to_name = edge.get('to')
            cost = edge.get('cost', 0.0)

            if from_name not in room_positions or to_name not in room_positions:
                continue

            start = room_positions[from_name]
            end = room_positions[to_name]

            # Line marker for edge
            line_marker = Marker()
            line_marker.header.frame_id = 'map'
            line_marker.header.stamp = self.get_clock().now().to_msg()
            line_marker.ns = 'edges'
            line_marker.id = self.next_id()
            line_marker.type = Marker.LINE_STRIP
            line_marker.action = Marker.ADD
            # Define two points for the line
            p0 = Point(x=start[0], y=start[1], z=0.0)
            p1 = Point(x=end[0], y=end[1], z=0.0)
            line_marker.points = [p0, p1]
            line_marker.scale.x = 0.05  # line width
            line_marker.color = ColorRGBA(r=0.0, g=0.0, b=0.0, a=1.0)  # black solid
            marker_array.markers.append(line_marker)

            # Text marker for edge cost at midpoint
            mid_x = (start[0] + end[0]) / 2.0
            mid_y = (start[1] + end[1]) / 2.0
            cost_text = Marker()
            cost_text.header.frame_id = 'map'
            cost_text.header.stamp = self.get_clock().now().to_msg()
            cost_text.ns = 'edge_labels'
            cost_text.id = self.next_id()
            cost_text.type = Marker.TEXT_VIEW_FACING
            cost_text.action = Marker.ADD
            cost_text.pose.position.x = mid_x
            cost_text.pose.position.y = mid_y
            cost_text.pose.position.z = 0.1
            cost_text.pose.orientation.w = 1.0
            cost_text.scale.z = 0.1
            cost_text.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            cost_text.text = f"{cost:.2f}m"
            marker_array.markers.append(cost_text)

        return marker_array

def main(args=None):
    rclpy.init(args=args)
    node = SemanticMapRvizPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()