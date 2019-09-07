#!usr/bin/env python3
import numpy as np
import random

class Node:
    def __init__(self, location, cost=0, children=[], parent=None):
        self.location = location
        self.cost = cost
        self.children = children
        self.parent = parent

    def add_child(self, node):
        self.children.append(node)

    def set_parent(self, parent):
        self.parent = parent

    def update_cost(self, cost):
        self.cost = cost

class RRT:
    def __init__(self, start, goal, limits, obstacles, threshold=0.1):
        self.start = start
        self.goal = goal
        self.dims = len(self.start)
        self.limits = limits
        self.obstacles = obstacles
        self.distance_threshold = threshold

        self.root = Node(list(start), cost=0)
        self.last_node = None
        self.node_list = [self.root]

        self.goal_eps = 0.1
        self.maximum_iterations = 1000

    def generate_random_point(self):
        location = []
        for i in range(self.dims):
            rand = random.random()
            scaling = self.limits[i][1] - self.limits[i][0]
            location.append(rand*scaling + self.limits[i][0])
        return location

    def check_collision(self, location):
        for obs in self.obstacles:
            dist = self.distance(obs[:-1], location)
            if dist <= obs[-1]:
                return True
        return False

    def check_path_collision(self, location_1, location_2):
        P1 = np.array(location_1)
        P2 = np.array(location_2)
        for obs in self.obstacles:
            Q = np.array(obs[:-1])
            t = np.dot(Q - P1, P2 - P1)/np.dot(P2 - P1, P2 - P1)
            if 0 <= t <= 1:
                return True
        return False

    def steer(self, start_location, end_location):
        dist = self.distance(start_location, end_location)
        if dist < self.distance_threshold:
            return end_location
        else:
            start_loc = np.array(start_location)
            direction = np.array(end_location) - start_loc
            return tuple(start_loc + self.distance_threshold * direction / dist)
        pass

    @staticmethod
    def distance(a, b):
        diff = np.array(a) - np.array(b)
        return np.sqrt(np.dot(diff, diff))

    def generate_path(self):
        iterations = 0
        found = False
        while iterations < self.maximum_iterations and not found:
            if random.random() < 0.1:
                point = self.goal
            else:
                while True:
                    point = self.generate_random_point()
                    if not self.check_collision(point):
                        break

            min_dist = np.inf
            min_node = None
            for n in self.node_list:
                d = self.distance(point, n.location)
                if d < min_dist and not self.check_path_collision(point, n.location):
                    min_dist = d
                    min_node = n

            if min_node is not None:
                point = self.steer(min_node.location, point)
                new_node = Node(point, parent=min_node, cost=min_node.cost + min_dist)
                min_node.add_child(new_node)
                self.node_list.append(new_node)

                goal_dist = self.distance(point, self.goal)
                if goal_dist < self.goal_eps:
                    if self.check_path_collision(self.goal, point):
                        continue
                    else:
                        goal_node = Node(self.goal, parent=new_node, cost=new_node.cost + goal_dist)
                        self.node_list.append(goal_node)
                        self.last_node = goal_node
                        found = True

                iterations += 1

        return found

    def retrieve_path(self):
        if self.last_node is None:
            found = self.generate_path()
            if not found:
                print("Could not find path in this trial. Try again.")
                return None
        path = []
        current_node = self.last_node
        while current_node is not None:
            path.append(current_node.location)
            current_node = current_node.parent
        return path[::-1]

    def retrieve_path_with_quick_smoothing(self, iterations=100):
        path = self.retrieve_path()
        if path is None:
            return path
        for i in range(iterations):
            p1 = random.choice(range(len(path)))
            p2 = random.choice(range(len(path)))
            if p1 == p2:
                continue
            if p1 > p2:
                temp = p2
                p2 = p1
                p1 = temp
            if not self.check_path_collision(path[p1], path[p2]):
                del path[p1+1:p2]
        return path

    def retrieve_path_with_smoothing(self):
        path = self.retrieve_path()
        if path is None:
            return path
        forward_iter = 0
        reverse_iter = len(path)-1
        while True:
            if forward_iter == len(path) - 1:
                break
            if forward_iter >= reverse_iter:
                forward_iter += 1
                reverse_iter = len(path) - 1
            if not self.check_path_collision(path[forward_iter], path[reverse_iter]):
                del path[forward_iter+1:reverse_iter]
                forward_iter = forward_iter + 1
                reverse_iter = len(path) - 1
            else:
                reverse_iter -= 1
        return path
