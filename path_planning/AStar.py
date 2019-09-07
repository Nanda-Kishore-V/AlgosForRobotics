#!usr/bin/env python3
import numpy as np
import itertools
import operator
import heapq

class AStar:
    def __init__(self, grid, start, goal):
        # Assumes grid is np.ndarray
        self.grid = grid
        dim = len(grid.shape)
        self.directions = [p for p in itertools.product([-1, 0, 1], repeat=dim) if any(p) != 0]
        self.direction_cost = []
        for i in range(len(self.directions)):
            cost = sum(tuple(map(operator.mul, self.directions[i], self.directions[i])))
            self.direction_cost.append(cost)
        self.path = None
        self.open_list = None
        self.closed_list = None
        self.reset(start, goal)

    def reset(self, start, goal):
        self.start = start
        self.goal = goal
        self.open_list = []
        self.closed_list = []
        self.path = dict()

    @staticmethod
    def distance(location_1, location_2):
        distance = 0
        for i in range(len(location_1)):
            distance += (location_1[i] - location_2[i])**2
        return np.sqrt(distance)

    def heuristic(self, location):
        return self.distance(location, self.goal)

    def check_validity(self, location):
        for i in range(len(location)):
            if location[i] < 0 or location[i] >= self.grid.shape[i]:
                return False
        if self.grid[location] != 0.0:
            return False
        return True

    def generate_path(self):
        found_path = False
        heapq.heappush(self.open_list, (self.heuristic(self.start), 0, self.start))
        self.path[self.start] = None
        while self.open_list:
            priority, cost_so_far, current_node = heapq.heappop(self.open_list)
            self.closed_list.append(current_node)
            if current_node == self.goal:
                found_path = True
                break
            for idx, direction in enumerate(self.directions):
                neighbor = tuple(map(operator.add, current_node, direction))
                if not self.check_validity(neighbor):
                    continue
                if neighbor in self.path.keys():
                    continue
                f = cost_so_far + self.direction_cost[idx]
                g = self.heuristic(neighbor)
                heapq.heappush(self.open_list, (f + g, f, neighbor))
                self.path[neighbor] = current_node
        if found_path == False:
            self.path = np.inf
        return found_path

    def retrieve_path(self):
        if not self.path:
            found_path = self.generate_path()
        if self.path == np.inf:
            print("No path found!")
            return []
        path = []
        current_node= self.goal
        while current_node != None:
            path.append(current_node)
            current_node = self.path[current_node]
        return path[::-1]
