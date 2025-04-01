#%%
# Name: Ayaan Khan
# Roll Number: 22i-0832
# Section: CS-K
# AI Assignment 02
# Question: 01

import math
import random
from collections import defaultdict
from copy import deepcopy

# Class to parse the graph file and store values
class Parser:
    def __init__(self, filename):
        self.filename = filename
        self.adjacency_list = defaultdict(set)
        self.heuristic_map = defaultdict(dict)
        self.edges = []
        self.vertices = set()

    def parse_graph(self):
        with open(self.filename, 'r') as f:
            next(f)  
            for line in f:
                line = line.strip()
                if not line:
                    continue  
                parts = line.split()
                if len(parts) != 3:
                    continue  
                source_str, dest_str, heuristic_str = parts
                u = int(source_str)
                v = int(dest_str)
                h = float(heuristic_str)
                self.adjacency_list[u].add(v)
                self.adjacency_list[v].add(u)
                self.edges.append((u, v))
                self.heuristic_map[u][v] = h
                self.heuristic_map[v][u] = h
                self.vertices.add(u)
                self.vertices.add(v)
        self.vertices = sorted(list(self.vertices))
        return self.adjacency_list, self.vertices, self.edges, self.heuristic_map

class State:
    def __init__(self, color_assignment=None):
        if color_assignment is None:
            color_assignment = {}
        self.color_assignment = color_assignment

    def copy_and_recolor(self, vertex, new_color):
        new_assignment = deepcopy(self.color_assignment)
        new_assignment[vertex] = new_color
        return State(color_assignment=new_assignment)

class Heuristic:
    def __init__(self, adjacency_list, 
                 distance_constraints=None, 
                 preassigned_colors=None, 
                 violation_weight=1000,
                 distinct_color_weight=300,
                 balance_weight=10):
        self.adjacency_list = adjacency_list
        self.distance_constraints = distance_constraints if distance_constraints else []
        self.preassigned_colors = preassigned_colors if preassigned_colors else {}
        self.violation_weight = violation_weight
        self.distinct_color_weight = distinct_color_weight
        self.balance_weight = balance_weight

# Combine penalties into a single cost (lower is better)
    def evaluate(self, state):
        violations = self.count_violations(state)
        balance_penalty = self.color_balance_score(state) * self.balance_weight
        distinct_color_penalty = self.distinct_color_count(state) * self.distinct_color_weight
        total_score = (violations * self.violation_weight) + balance_penalty + distinct_color_penalty
        return total_score

    def count_violations(self, state):
        violations = 0

        # 1) Adjacency
        for v, neighbors in self.adjacency_list.items():
            v_color = state.color_assignment[v]
            for nbr in neighbors:
                if nbr > v: 
                    if state.color_assignment[nbr] == v_color:
                        violations += 1
        # 2) Distance constraints
        for (u, w) in self.distance_constraints:
            if state.color_assignment[u] == state.color_assignment[w]:
                violations += 1
        # 3) Preassigned colors
        for vertex, required_color in self.preassigned_colors.items():
            if state.color_assignment[vertex] != required_color:
                violations += 10 
        return violations

    # Uses Standard Deviation
    def color_balance_score(self, state):
        used_colors = list(set(state.color_assignment.values()))
        if not used_colors:
            return 0
        usage_counts = []
        for color in used_colors:
            count = sum(1 for c in state.color_assignment.values() if c == color)
            usage_counts.append(count)
        mean_usage = sum(usage_counts) / float(len(usage_counts))
        variance = sum((count - mean_usage) ** 2 for count in usage_counts) / float(len(usage_counts))
        return math.sqrt(variance)

    def distinct_color_count(self, state):
        return len(set(state.color_assignment.values()))

class LocalBeamSearch:
    def __init__(self, adjacency_list, vertices, beam_size=3, max_iterations=100, 
                 preassigned_colors=None, distance_constraints=None):
        self.adjacency_list = adjacency_list
        self.vertices = vertices
        self.beam_size = beam_size
        self.max_iterations = max_iterations
        self.preassigned_colors = preassigned_colors if preassigned_colors else {}
        self.distance_constraints = distance_constraints if distance_constraints else []

        self.heuristic = Heuristic(
            adjacency_list=self.adjacency_list,
            distance_constraints=self.distance_constraints,
            preassigned_colors=self.preassigned_colors,
            violation_weight=1000,
            distinct_color_weight=300,
            balance_weight=10
        )

    def run(self):
        current_beam = self.generate_initial_states()
        best_state = None
        best_score = float('inf')
        for iteration in range(self.max_iterations):
            scored_beam = []
            for state in current_beam:
                score = self.heuristic.evaluate(state)
                if score < best_score:
                    best_score = score
                    best_state = state
                scored_beam.append((score, state))
            scored_beam.sort(key=lambda x: x[0])
            # Early exit if the best state has zero violations.
            top_state = scored_beam[0][1]
            if self.heuristic.count_violations(top_state) == 0:
                best_state = top_state
                best_score = scored_beam[0][0]
                break
            successor_pool = []
            for (_, st) in scored_beam:
                successors = self.generate_successors(st)
                for succ in successors:
                    succ_score = self.heuristic.evaluate(succ)
                    successor_pool.append((succ_score, succ))
            combined = scored_beam + successor_pool
            combined.sort(key=lambda x: x[0])
            current_beam = [x[1] for x in combined[:self.beam_size]]
        return best_state, best_score

    def generate_initial_states(self):
        states = []
        degrees = self.get_degrees()
        for _ in range(self.beam_size):
            color_assignment = {}
            for v in self.vertices:
                if v in self.preassigned_colors:
                    color_assignment[v] = self.preassigned_colors[v]
                else:
                    color_assignment[v] = None
            for v in sorted(self.vertices, key=lambda x: -degrees[x]):
                if color_assignment[v] is None:
                    neighbor_colors = {color_assignment[nbr] for nbr in self.adjacency_list[v] if color_assignment[nbr] is not None}
                    candidate = 0
                    while candidate in neighbor_colors:
                        candidate += 1
                    color_assignment[v] = candidate
            states.append(State(color_assignment=color_assignment))
        return states

    def generate_successors(self, state):
        successors = []
        degrees = self.get_degrees()
        sorted_vertices = sorted(self.vertices, key=lambda v: -degrees[v])
        max_successors_per_vertex = 3
        total_successors = 0
        max_total_successors = 10
        for v in sorted_vertices:
            if v in self.preassigned_colors:
                continue
            current_color = state.color_assignment[v]
            attempted_colors = 0
            max_color = max(state.color_assignment.values())
            for new_color in range(max_color + 2):
                if new_color == current_color:
                    continue
                new_state = state.copy_and_recolor(v, new_color)
                successors.append(new_state)
                attempted_colors += 1
                total_successors += 1
                if attempted_colors >= max_successors_per_vertex:
                    break
                if total_successors >= max_total_successors:
                    break
            if total_successors >= max_total_successors:
                break
        return successors

    def get_degrees(self):
        degrees = {}
        for v, neighbors in self.adjacency_list.items():
            degrees[v] = len(neighbors)
        return degrees

def main():
    graph_file = 'hypercube_dataset.txt' 
    parser = Parser(graph_file)
    adjacency_list, vertices, edges, heuristic_map = parser.parse_graph()

    preassigned_colors = {
        0: 0, 
        1: 1   
    }
    distance_constraints = [
        (1, 3),  
    ]
    beam_size = 50
    max_iterations = 5000

    lbs = LocalBeamSearch(
        adjacency_list=adjacency_list,
        vertices=vertices,
        beam_size=beam_size,
        max_iterations=max_iterations,
        preassigned_colors=preassigned_colors,
        distance_constraints=distance_constraints
    )

    best_state, best_score = lbs.run()

    print("=== Local Beam Search - Graph Coloring Results ===")
    print(f"Best Score: {best_score}")
    print("Color Assignment (vertex -> color):")
    for v in sorted(best_state.color_assignment.keys()):
        print(f"  Vertex {v} -> Color {best_state.color_assignment[v]}")
    final_violations = lbs.heuristic.count_violations(best_state)
    balance_val = lbs.heuristic.color_balance_score(best_state)
    distinct_colors_used = len(set(best_state.color_assignment.values()))
    print(f"Final Constraint Violations: {final_violations}")
    print(f"Distinct Colors Used: {distinct_colors_used}")
    print(f"Standard Deviation of Colours Used (Balance Score): {balance_val}")

if __name__ == '__main__':
    main()

# %%
