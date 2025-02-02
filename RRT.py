import numpy as np
from scipy import spatial
import math
from math import atan2, pi
import glob
import os
import sys
import carla

# Append the CARLA egg file to the path
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    import carla
except IndexError:
    pass

# -----------------------------------------------------------------------------
# Node class for tree nodes
# -----------------------------------------------------------------------------
class Node:
    """
    Node represents a point in the RRT tree.

    Attributes:
        x (float): x-coordinate in the world.
        y (float): y-coordinate in the world.
        waypoint: The CARLA waypoint associated with the node.
        parent (Node): The parent node in the tree.
        cost (float): The cost (edge weight) from the parent node.
    """
    def __init__(self, waypoint):
        self.x = waypoint.transform.location.x
        self.y = waypoint.transform.location.y
        self.waypoint = waypoint
        self.parent = None
        self.cost = 0.0

# -----------------------------------------------------------------------------
# RRT class (with RRT, RRT* and Informed RRT* search methods)
# -----------------------------------------------------------------------------
class RRT:
    """
    RRT implements the Rapidly-exploring Random Tree (RRT) algorithm and its variants
    (RRT* and Informed RRT*) for path planning in the CARLA world.

    Attributes:
        world: The CARLA world wrapper.
        obstacles: A list of obstacle transforms (vehicles) used for collision checking.
        map: The CARLA map.
        start (Node): The start node for planning.
        goal (Node): The goal node for planning.
        vertices (list): List of nodes in the tree.
        found (bool): Flag indicating whether a path to the goal has been found.
        path (list): Final path from start to goal as a list of nodes.
        size_x (float): The sampling range in x-direction.
        size_y (float): The sampling range in y-direction.
    """
    def __init__(self, world, goal, obstacles):
        """
        Initialize the RRT with the world, goal, and obstacles.

        Args:
            world: The CARLA world instance.
            goal: The CARLA waypoint for the goal.
            obstacles: A list of CARLA transforms representing obstacles.
        """
        self.world = world
        self.obstacles = obstacles
        self.map = world.map
        self.world.carla_world.tick()
        trans = self.world.ego_vehicle.get_transform()
        x_ego, y_ego = trans.location.x, trans.location.y
        start = self.map.get_waypoint(carla.Location(x=x_ego, y=y_ego), project_to_road=False)
        self.size_x = 10    # X-range for sampling
        self.size_y = 100   # Y-range for sampling
        self.start = Node(start)
        self.goal = Node(goal)
        self.vertices = []
        self.found = False
        self.path = []

    def init_map(self):
        """
        Initialize the search by resetting the vertices list and the found flag.
        """
        self.found = False
        self.vertices = [self.start]

    def dis(self, node1, node2):
        """
        Compute the Euclidean distance between two nodes.

        Args:
            node1 (Node): The first node.
            node2 (Node): The second node.

        Returns:
            float: Euclidean distance between node1 and node2.
        """
        return np.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)

    def check_collision(self, node1, node2):
        """
        Check if the path between node1 and node2 collides with any obstacles.

        The function samples integer points along the straight line from node1 to node2
        and determines whether any point is too close to an obstacle based on a tolerance
        distance computed from vehicle dimensions.

        Args:
            node1 (Node): The starting node.
            node2 (Node): The ending node.

        Returns:
            bool: True if a collision is detected, False otherwise.
        """
        points_between = zip(np.linspace(node1.x, node2.x, dtype=int),
                             np.linspace(node1.y, node2.y, dtype=int))
        yaw_node = atan2(node2.y - node1.y, node2.x - node1.x)
        for point in points_between:
            for vehicle in self.obstacles:
                yaw_obstacle = vehicle.rotation.yaw * pi / 180
                x_obs, y_obs = vehicle.location.x, vehicle.location.y
                dist = np.sqrt((x_obs - point[0]) ** 2 + (y_obs - point[1]) ** 2)
                yaw = atan2(y_obs - point[1], x_obs - point[0])
                # Define vehicle dimensions and tolerance angles
                r_big, r_small = 2.7, 1.66
                theta = 0.646  # Approximately 37 degrees in radians
                if (-theta < (yaw - yaw_node) < theta or 
                    (pi - theta) < (yaw - yaw_node) <= pi or 
                    -pi <= (yaw - yaw_node) <= (-pi + theta)):
                    min_dist = 2 * r_big if (-theta < (yaw - yaw_obstacle) < theta or 
                                              (pi - theta) < (yaw - yaw_obstacle) <= pi or 
                                              -pi <= (yaw - yaw_obstacle) <= (-pi + theta)) else (r_big + r_small)
                else:
                    min_dist = (r_big + r_small) if (-theta < (yaw - yaw_obstacle) < theta or 
                                                     (pi - theta) < (yaw - yaw_obstacle) <= pi or 
                                                     -pi <= (yaw - yaw_obstacle) <= (-pi + theta)) else (2 * r_small)
                if dist <= min_dist:
                    return True
        return False

    def get_new_point(self, goal_bias):
        """
        Generate a new random point with a bias towards the goal.

        Args:
            goal_bias (float): Probability of choosing the goal directly.

        Returns:
            list: A point as [x, y] coordinates.
        """
        if np.random.random() < goal_bias:
            return [self.goal.x, self.goal.y]
        else:
            return [np.random.randint(self.start.x - self.size_x, self.start.x + self.size_x),
                    np.random.randint(self.start.y - self.size_y, self.start.y + self.size_y)]

    def get_random_ball(self):
        """
        Sample a random point from a unit ball (for informed sampling).

        Returns:
            numpy.matrix: A 1x2 matrix with coordinates sampled uniformly from the unit ball.
        """
        u, v = np.random.random(), np.random.random()
        r = u ** 0.5
        theta = 2 * math.pi * v
        return np.mat([r * math.cos(theta), r * math.sin(theta)])

    def get_new_point_in_ellipsoid(self, goal_bias, c_best):
        """
        Generate a new point within an ellipsoid defined by the start, goal, and the
        current best path length (for informed RRT* sampling).

        Args:
            goal_bias (float): Probability of selecting the goal.
            c_best (float): Current best path length.

        Returns:
            list: A point as [x, y] coordinates.
        """
        if np.random.random() < goal_bias:
            return [self.goal.x, self.goal.y]
        else:
            c_min = self.dis(self.start, self.goal)
            x_center = np.mat([(self.goal.x + self.start.x) / 2, (self.goal.y + self.start.y) / 2])
            theta = math.atan2(self.goal.y - self.start.y, self.goal.x - self.start.x)
            C = np.mat([[math.cos(theta), math.sin(theta)], 
                        [-math.sin(theta), math.cos(theta)]])
            L = np.mat([[c_best / 2, 0], [0, ((c_best ** 2 - c_min ** 2) ** 0.5) / 2]])
            Xball = self.get_random_ball()
            x_rand = C * L * Xball.transpose() + x_center.transpose()
            x_rand = np.array(x_rand)
            return [x_rand.transpose()[0][0], x_rand.transpose()[0][1]]

    def get_nearest_node(self, point):
        """
        Find the nearest node in the current tree to a given point.

        Args:
            point (list): A point as [x, y] coordinates.

        Returns:
            Node: The nearest node from the tree.
        """
        samples = [[v.x, v.y] for v in self.vertices]
        kdtree = spatial.cKDTree(samples)
        _, ind = kdtree.query(point)
        return self.vertices[ind]

    def sample(self, goal_bias=0.05, c_best=0):
        """
        Sample a new point for the tree expansion. If a best path exists (c_best > 0),
        sample within an ellipsoid; otherwise, sample uniformly.

        Args:
            goal_bias (float): The probability of choosing the goal.
            c_best (float): The current best path length.

        Returns:
            list: A point as [x, y] coordinates.
        """
        if c_best <= 0:
            return self.get_new_point(goal_bias)
        else:
            return self.get_new_point_in_ellipsoid(goal_bias, c_best)

    def extend(self, new_point, extend_dis=5):
        """
        Extend the tree toward a new point by a fixed extension distance.

        Args:
            new_point (list): The target point as [x, y].
            extend_dis (float): The extension distance.

        Returns:
            Node or None: The newly created node if extension is successful, else None.
        """
        new_waypoint = self.map.get_waypoint(carla.Location(x=new_point[0], y=new_point[1]), 
                                              project_to_road=True)
        new_point = [new_waypoint.transform.location.x, new_waypoint.transform.location.y]
        nearest_node = self.get_nearest_node(new_point)
        slope = math.atan2(new_point[1] - nearest_node.y, new_point[0] - nearest_node.x)
        new_x = nearest_node.x + extend_dis * math.cos(slope)
        new_y = nearest_node.y + extend_dis * math.sin(slope)
        new_waypoint = self.map.get_waypoint(carla.Location(x=new_x, y=new_y), 
                                              project_to_road=True)
        new_node = Node(new_waypoint)
        if ((self.start.x - self.size_x <= new_node.x < self.start.x + self.size_x) and
            (self.start.y - self.size_y <= new_node.y < self.start.y + self.size_y) and
            not self.check_collision(nearest_node, new_node)):
            new_node.parent = nearest_node
            new_node.cost = extend_dis
            self.vertices.append(new_node)
            if not self.found:
                d = self.dis(new_node, self.goal)
                if d < extend_dis:
                    self.goal.cost = d
                    self.goal.parent = new_node
                    self.vertices.append(self.goal)
                    self.found = True
            return new_node
        else:
            return None

    def get_neighbors(self, new_node, neighbor_size):
        """
        Find all nodes within a specified radius (neighbor_size) of new_node.

        Args:
            new_node (Node): The node for which to find neighbors.
            neighbor_size (float): The radius for neighbor search.

        Returns:
            list: A list of neighboring nodes.
        """
        samples = [[v.x, v.y] for v in self.vertices]
        kdtree = spatial.cKDTree(samples)
        indices = kdtree.query_ball_point([new_node.x, new_node.y], neighbor_size)
        neighbors = [self.vertices[i] for i in indices]
        if new_node in neighbors:
            neighbors.remove(new_node)
        return neighbors

    def generate_path(self, start_node, end_node):
        """
        Backtrack from end_node to start_node using parent pointers to generate the path.

        Args:
            start_node (Node): The start node.
            end_node (Node): The goal node.

        Returns:
            list: The path as a list of nodes from start to goal.
        """
        path = []
        curr_node = end_node
        while start_node.x != curr_node.x or start_node.y != curr_node.y:
            path.append(curr_node)
            if curr_node.parent is None:
                print("Invalid Path")
                return []
            curr_node = curr_node.parent
        return path[::-1]

    def path_cost(self, start_node, end_node):
        """
        Compute the cost of the path from start_node to end_node by summing edge costs.

        Args:
            start_node (Node): The start node.
            end_node (Node): The goal node.

        Returns:
            float: The total path cost.
        """
        cost = 0
        curr_node = end_node
        while start_node.x != curr_node.x or start_node.y != curr_node.y:
            if curr_node.parent is None:
                print("Invalid Path")
                return 0
            cost += curr_node.cost
            curr_node = curr_node.parent
        return cost

    def rewire(self, new_node, neighbors):
        """
        Rewire the tree: adjust the parent of new_node and its neighbors if a lower cost
        connection is possible.

        Args:
            new_node (Node): The newly added node.
            neighbors (list): List of neighboring nodes.
        """
        if not neighbors:
            return
        distances = [self.dis(new_node, node) for node in neighbors]
        costs = [d + self.path_cost(self.start, neighbors[i]) for i, d in enumerate(distances)]
        indices = np.argsort(np.array(costs))
        for i in indices:
            if not self.check_collision(new_node, neighbors[i]):
                new_node.parent = neighbors[i]
                new_node.cost = distances[i]
                break
        for i, node in enumerate(neighbors):
            new_cost = self.path_cost(self.start, new_node) + distances[i]
            if self.path_cost(self.start, node) > new_cost and not self.check_collision(node, new_node):
                node.parent = new_node
                node.cost = distances[i]

    def RRT(self, n_pts=1000):
        """
        Run the basic RRT algorithm to find a path from start to goal.

        Args:
            n_pts (int): Maximum number of points to sample.
        """
        self.init_map()
        for _ in range(n_pts):
            new_point = self.sample(0.05, 0)
            self.extend(new_point, 10)
            if self.found:
                break
        if self.found:
            steps = len(self.vertices) - 2
            length = self.path_cost(self.start, self.goal)
            self.path = self.generate_path(self.start, self.goal)
            print("It took %d nodes to find the path" % steps)
            print("The path length is %.2f" % length)
        else:
            print("No path found")

    def RRT_star(self, n_pts=1000, neighbor_size=10):
        """
        Run the RRT* algorithm, which includes rewiring for an improved path.

        Args:
            n_pts (int): Maximum number of points to sample.
            neighbor_size (float): The neighbor radius for rewiring.
        """
        self.init_map()
        for _ in range(n_pts):
            new_point = self.sample(0.05, 0)
            new_node = self.extend(new_point, 8)
            if new_node is not None:
                neighbors = self.get_neighbors(new_node, neighbor_size)
                self.rewire(new_node, neighbors)
        if self.found:
            steps = len(self.vertices) - 2
            length = self.path_cost(self.start, self.goal)
            self.path = self.generate_path(self.start, self.goal)
            print("It took %d nodes to find the path" % steps)
            print("The path length is %.2f" % length)
        else:
            print("No path found")

    def informed_RRT_star(self, n_pts=1000, neighbor_size=20):
        """
        Run the Informed RRT* algorithm which restricts sampling to an ellipsoidal region
        once a solution is found.

        Args:
            n_pts (int): Maximum number of points to sample.
            neighbor_size (float): The neighbor radius for rewiring.
        """
        self.init_map()
        for _ in range(n_pts):
            c_best = self.path_cost(self.start, self.goal) if self.found else 0
            new_point = self.sample(0.05, c_best)
            new_node = self.extend(new_point, 10)
            if new_node is not None:
                neighbors = self.get_neighbors(new_node, neighbor_size)
                self.rewire(new_node, neighbors)
        if self.found:
            steps = len(self.vertices) - 2
            length = self.path_cost(self.start, self.goal)
            print("It took %d nodes to find the path" % steps)
            print("The path length is %.2f" % length)
        else:
            print("No path found")
