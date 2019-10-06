#!/usr/bin/env python3
import numpy as np
import itertools
import operator
import heapq

class LPAStar:
    def __init__(self, grid, start, goal):
        self.grid = grid
        dim = len(grid.shape)
        self.directions = [p for p in itertools.product([-1, 0, 1], repeat=dim) if any(p) != 0]
        self.direction_cost = []
        for i in range(len(self.directions)):
            cost = sum(tuple(map(operator.mul, self.directions[i], self.directions[i])))
            self.direction_cost.append(cost)
        self.path = None
        self.reset(start, goal)

    def reset(self, start, goal):
        self.start = start
        self.goal = goal
        self.g = {}
        self.rhs = {}
        self.path = {}
        self.initialize()
        self.compute_shortest_path()

    @staticmethod
    def distance(location_1, location_2):
        distance = 0
        for i in range(len(location_1)):
            distance += (location_1[i] - location_2[i])**2
        return np.sqrt(distance)

    def heuristic(self, location):
        return self.distance(location, self.goal)

    def check_location(self, location):
        for i in range(len(location)):
            if location[i] <0 or location[i] >= self.grid.shape[i]:
                return False
        return True

    def check_validity(self, location):
        return self.check_location(location)
        if self.grid[location] != 0.0:
            return False
        return True

    def initialize(self):
        self.queue = []
        self.rhs[self.start] = 0
        heapq.heappush(self.queue, (self.calculate_key(self.start), self.start))

    def calculate_key(self, node):
        k2 = min(self.g.get(node, np.inf), self.rhs.get(node, np.inf))
        return [k2 + self.heuristic(node), k2]

    def update_vertex(self, node):
        if self.check_location(node):
            if self.grid[node] == 1:
                self.g[node] = np.inf
                self.rhs[node] = np.inf
                return
        if node != self.start:
            min_value = np.inf
            for idx, direction in enumerate(self.directions):
                neighbor = tuple(map(operator.add, node, direction))
                if self.check_location(neighbor):
                    if self.grid[neighbor] != 0:
                        current_value = np.inf
                    else:
                        current_value = self.g.get(neighbor, np.inf) + self.direction_cost[idx]
                    if min_value > current_value:
                        min_value = current_value
            self.rhs[node] = min_value
        self.queue = [i for i in self.queue if i[1] != node]
        heapq.heapify(self.queue)
        if self.g.get(node, np.inf) != self.rhs.get(node, np.inf):
            heapq.heappush(self.queue, (self.calculate_key(node), node))

    def compute_shortest_path(self):
        while heapq.nsmallest(1, self.queue)[0][0] < self.calculate_key(self.goal) or \
                self.rhs.get(self.goal, np.inf) != self.g.get(self.goal, np.inf):
            k, current_node = heapq.heappop(self.queue)
            if self.g.get(current_node, np.inf) > self.rhs.get(current_node, np.inf):
                self.g[current_node] = self.rhs[current_node]
                for direction in self.directions:
                    neighbor = tuple(map(operator.add, current_node, direction))
                    if self.check_validity(neighbor):
                        self.update_vertex(neighbor)
            else:
                self.g[current_node] = np.inf
                for direction in self.directions:
                    neighbor = tuple(map(operator.add, current_node, direction))
                    if self.check_validity(neighbor):
                        self.update_vertex(neighbor)
                self.update_vertex(current_node)

    def generate_path(self, grid):
        diff_grid = self.grid - grid
        self.grid = grid
        I, J = np.nonzero(diff_grid)
        for i, j in zip(I, J):
            self.update_vertex((i,j))

            for direction in self.directions:
                neighbor = tuple(map(operator.add, (i, j), direction))
                if self.check_location(neighbor):
                    if self.grid[neighbor] != 0:
                        self.g[neighbor] = np.inf
                        self.rhs[neighbor] = np.inf
                    else:
                        self.update_vertex(neighbor)

        self.compute_shortest_path()

    def retrieve_path(self):
        self.path = [self.goal]
        node = self.goal
        while node != self.start:
            min_rhs = self.rhs.get(node, np.inf)
            min_neighbor = None
            for direction in self.directions:
                neighbor = tuple(map(operator.add, node, direction))
                if self.rhs.get(neighbor, np.inf) < min_rhs:
                    min_rhs = self.rhs.get(neighbor, np.inf)
                    min_neighbor = neighbor
            if min_neighbor is None:
                print("Failed!")
                return -1
            self.path.append(min_neighbor)
            node = min_neighbor
        return self.path[::-1]
