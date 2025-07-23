import yaml
import os
import math
import heapq
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum

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

class NodeType(Enum):
    ROOM = "room"
    OBJECT = "object"

@dataclass
class Position:
    x: float
    y: float
    theta: float = 0.0
    
    def distance_to(self, other: 'Position') -> float:
        """Calculate Euclidean distance to another position"""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def __hash__(self):
        return hash((self.x, self.y, self.theta))

@dataclass
class GraphNode:
    id: str
    position: Position
    node_type: NodeType
    features: List[Dict] = field(default_factory=list)
    
    def __hash__(self):
        return hash(self.id)

class DStarLitePlanner:
    """D* Lite path planning algorithm for topological navigation"""
    
    def __init__(self, memory_file: Optional[str] = None):
        if memory_file is None:
            # Use workspace root to construct the path
            workspace_root = find_workspace_root()
            self.memory_file = os.path.join(workspace_root, "src", "memory.yaml")
        else:
            self.memory_file = memory_file
            
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: Dict[Tuple[str, str], float] = {}
        self.start_node: Optional[str] = None
        self.goal_node: Optional[str] = None
        
        # Load memory data
        self.load_memory_data()
        
    def load_memory_data(self):
        """Load room and object data from memory.yaml"""
        try:
            with open(self.memory_file, 'r') as file:
                data = yaml.safe_load(file)
                
            # Process nodes (rooms)
            for node_data in data.get('nodes', []):
                room_name = node_data.get('name', 'unknown_room')
                pose = node_data.get('pose', [0.0, 0.0, 0.0])
                features = node_data.get('features', [])
                
                # Create room node
                room_position = Position(pose[0], pose[1], pose[2])
                room_node = GraphNode(
                    id=room_name,
                    position=room_position,
                    node_type=NodeType.ROOM,
                    features=features
                )
                self.nodes[room_name] = room_node
                print(f"Loaded room: {room_name}")
                
                # Create object nodes from features
                for i, feature in enumerate(features):
                    if 'object' in feature and 'Coordinate relative to the world frame' in feature:
                        coords = feature['Coordinate relative to the world frame']
                        if len(coords) >= 2:
                            object_name = f"{room_name}_{feature['object']}_{i}"
                            object_position = Position(coords[0], coords[1], 0.0)
                            object_node = GraphNode(
                                id=object_name,
                                position=object_position,
                                node_type=NodeType.OBJECT,
                                features=[feature]
                            )
                            self.nodes[object_name] = object_node
                            print(f"  Loaded object: {object_name}")
                            
        except FileNotFoundError:
            print(f"Warning: Memory file {self.memory_file} not found")
        except Exception as e:
            print(f"Error loading memory data: {e}")
        
        print(f"Total nodes loaded: {len(self.nodes)}")
        print(f"Available nodes: {list(self.nodes.keys())}")
    
    def calculate_edge_cost(self, node1: GraphNode, node2: GraphNode) -> float:
        """Calculate cost between two nodes"""
        base_distance = node1.position.distance_to(node2.position)
        
        # Add penalties based on node types
        penalty = 0.0
        
        # Higher cost for moving between different room types
        if (node1.node_type == NodeType.ROOM and node2.node_type == NodeType.ROOM):
            penalty += 5.0  # Room-to-room transition cost
            
        # Lower cost for moving to objects within the same room
        if (node1.node_type == NodeType.OBJECT and node2.node_type == NodeType.OBJECT):
            # Check if they're in the same room
            room1 = node1.id.split('_')[0]
            room2 = node2.id.split('_')[0]
            if room1 == room2:
                penalty -= 1.0  # Bonus for same room objects
                
        return base_distance + penalty
    
    def build_graph(self):
        """Build the graph with edges between all nodes"""
        node_ids = list(self.nodes.keys())
        
        for i, node1_id in enumerate(node_ids):
            for j, node2_id in enumerate(node_ids):
                if i != j:
                    node1 = self.nodes[node1_id]
                    node2 = self.nodes[node2_id]
                    cost = self.calculate_edge_cost(node1, node2)
                    self.edges[(node1_id, node2_id)] = cost
    
    def get_neighbors(self, node_id: str) -> List[str]:
        """Get all neighbors of a node"""
        neighbors = []
        for (src, dst), cost in self.edges.items():
            if src == node_id:
                neighbors.append(dst)
        return neighbors
    
    def calculate_key(self, node_id: str) -> Tuple[float, float]:
        """Calculate key for priority queue"""
        g_val = self.g.get(node_id, float('inf'))
        rhs_val = self.rhs.get(node_id, float('inf'))
        h_val = self.heuristic(node_id)
        
        return (min(g_val, rhs_val) + h_val + self.km, min(g_val, rhs_val))
    
    def heuristic(self, node_id: str) -> float:
        """Heuristic function (distance to goal)"""
        if self.goal_node is None or node_id not in self.nodes:
            return 0.0
        
        current_pos = self.nodes[node_id].position
        goal_pos = self.nodes[self.goal_node].position
        return current_pos.distance_to(goal_pos)
    
    def update_vertex(self, node_id: str):
        """Update vertex in D* Lite algorithm"""
        if node_id != self.start_node:
            min_rhs = float('inf')
            for neighbor in self.get_neighbors(node_id):
                if neighbor in self.g:
                    cost = self.edges.get((neighbor, node_id), float('inf'))
                    min_rhs = min(min_rhs, self.g[neighbor] + cost)
            self.rhs[node_id] = min_rhs
        
        # Remove from queue if present
        self.queue = [(k1, k2, n) for k1, k2, n in self.queue if n != node_id]
        
        if self.g.get(node_id, float('inf')) != self.rhs.get(node_id, float('inf')):
            key = self.calculate_key(node_id)
            heapq.heappush(self.queue, (key[0], key[1], node_id))
    
    def compute_shortest_path(self):
        """Compute shortest path using D* Lite"""
        if self.start_node is None:
            return
            
        while self.queue and (
            self.queue[0][:2] < self.calculate_key(self.start_node) or
            self.rhs.get(self.start_node, float('inf')) > self.g.get(self.start_node, float('inf'))
        ):
            k_old = self.queue[0][:2]
            u = heapq.heappop(self.queue)[2]
            
            if k_old < self.calculate_key(u):
                heapq.heappush(self.queue, (*self.calculate_key(u), u))
            elif self.g.get(u, float('inf')) > self.rhs.get(u, float('inf')):
                self.g[u] = self.rhs[u]
                for neighbor in self.get_neighbors(u):
                    self.update_vertex(neighbor)
            else:
                self.g[u] = float('inf')
                self.update_vertex(u)
                for neighbor in self.get_neighbors(u):
                    self.update_vertex(neighbor)
    
    def plan_path(self, start_location: str, goal_location: str) -> List[str]:
        """
        Plan a path from start to goal location
        
        Args:
            start_location: Name of starting room or object
            goal_location: Name of goal room or object
            
        Returns:
            List of node IDs representing the path
        """
        # Find the actual node IDs
        start_node_id = self.find_node_by_name(start_location)
        goal_node_id = self.find_node_by_name(goal_location)
        
        if start_node_id is None or goal_node_id is None:
            print(f"Could not find nodes: start={start_location}, goal={goal_location}")
            return []
        
        self.start_node = start_node_id
        self.goal_node = goal_node_id
        
        # Initialize D* Lite
        self.rhs = {goal_node_id: 0.0}
        self.g = {goal_node_id: float('inf')}
        self.queue = []
        self.km = 0.0
        
        # Build graph if not already built
        if not self.edges:
            self.build_graph()
        
        # Initialize goal node
        self.update_vertex(goal_node_id)
        
        # Compute shortest path
        self.compute_shortest_path()
        
        # Extract path
        path = self.extract_path()
        return path
    
    def find_node_by_name(self, location_name: str) -> Optional[str]:
        """Find node ID by location name (room or object)"""
        # First try exact match
        if location_name in self.nodes:
            return location_name
        
        # Try partial matches
        for node_id in self.nodes:
            if location_name.lower() in node_id.lower():
                return node_id
        
        # Try matching by object type
        for node_id, node in self.nodes.items():
            if node.node_type == NodeType.OBJECT and node.features:
                for feature in node.features:
                    if 'object' in feature and location_name.lower() in feature['object'].lower():
                        return node_id
        
        return None
    
    def extract_path(self) -> List[str]:
        """Extract the path from start to goal"""
        if self.start_node is None or self.start_node not in self.g or self.g[self.start_node] == float('inf'):
            return []
        
        path = [self.start_node]
        current = self.start_node
        
        while current != self.goal_node:
            best_neighbor = None
            best_cost = float('inf')
            
            for neighbor in self.get_neighbors(current):
                if neighbor in self.g:
                    cost = self.edges.get((current, neighbor), float('inf'))
                    total_cost = self.g[neighbor] + cost
                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_neighbor = neighbor
            
            if best_neighbor is None:
                break
                
            path.append(best_neighbor)
            current = best_neighbor
        
        return path
    
    def get_path_cost(self, path: List[str]) -> float:
        """Calculate total cost of a path"""
        if len(path) < 2:
            return 0.0
        
        total_cost = 0.0
        for i in range(len(path) - 1):
            cost = self.edges.get((path[i], path[i + 1]), float('inf'))
            total_cost += cost
        
        return total_cost
    
    def print_path_info(self, path: List[str]):
        """Print detailed information about the planned path"""
        if not path:
            print("No path found!")
            return
        
        print(f"\n=== Motion Planning Results ===")
        print(f"Path length: {len(path)} nodes")
        print(f"Total cost: {self.get_path_cost(path):.2f} units")
        print(f"\nPath:")
        
        for i, node_id in enumerate(path):
            node = self.nodes[node_id]
            if node.node_type == NodeType.ROOM:
                print(f"  {i+1}. Room: {node_id}")
            else:
                print(f"  {i+1}. Object: {node_id}")
                if node.features:
                    obj_name = node.features[0].get('object', 'unknown')
                    print(f"     - Object type: {obj_name}")
            
            print(f"     - Position: ({node.position.x:.2f}, {node.position.y:.2f}, {node.position.theta:.2f})")
        
        print("=" * 30)


def plan_motion_to_location(start_location: str, goal_location: str, memory_file: Optional[str] = None) -> List[str]:
    """
    Main function to plan motion from start to goal location
    
    Args:
        start_location: Starting room or object name
        goal_location: Goal room or object name
        memory_file: Path to memory.yaml file
        
    Returns:
        List of node IDs representing the optimal path
    """
    planner = DStarLitePlanner(memory_file)
    path = planner.plan_path(start_location, goal_location)
    
    if path:
        planner.print_path_info(path)
    else:
        print(f"No path found from {start_location} to {goal_location}")
    
    return path


# Example usage
if __name__ == "__main__":
    # Example: Plan path from robotics lab to workbench
    path = plan_motion_to_location("robotics lab", "workbench")
    
    # Example: Plan path to a specific object
    # path = plan_motion_to_location("robotics lab", "robotics lab_workbench_0") 