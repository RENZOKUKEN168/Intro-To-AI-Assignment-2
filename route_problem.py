"""Problem, search-tree nodes and the assignment's plain-text input format.

The course's Problem/Node concepts are implemented here without importing
course algorithms or third-party search libraries.
"""
from dataclasses import dataclass, field
from math import hypot, isfinite
from pathlib import Path
import re
from typing import Optional


class RouteProblem:
    """A finite, directed graph with nonnegative costs and at least one goal."""

    def __init__(self, graph, origin, destinations, coordinates):
        if not coordinates:
            raise ValueError("Provide nodes and at least one destination.")
        self.coordinates = dict(coordinates)
        self.graph = {state: dict(graph.get(state, {})) for state in coordinates}
        self.initial = origin
        self.goals = frozenset(destinations)
        if not self.goals:
            raise ValueError("At least one destination is required.")
        states = set(coordinates)
        if any(type(state) is not int for state in states):
            raise ValueError("Node IDs must be integers.")
        references = set(graph) | {v for edges in graph.values() for v in edges}
        references |= {origin} | set(self.goals)
        if not references <= states:
            raise ValueError("An edge, origin or goal references an undefined node.")
        for point in coordinates.values():
            if len(point) != 2 or not all(isfinite(value) for value in point):
                raise ValueError("Coordinates must be finite (x,y) pairs.")
        for edges in self.graph.values():
            if any(not isfinite(cost) or cost < 0 for cost in edges.values()):
                raise ValueError("Edge costs must be finite and nonnegative.")

    def actions(self, state):
        return sorted(self.graph[state])

    def result(self, state, action):
        if action not in self.graph[state]:
            raise ValueError("This directed edge does not exist.")
        return action

    def goal_test(self, state):
        return state in self.goals

    def path_cost(self, cost_so_far, state, action, next_state):
        value = cost_so_far + self.graph[state][next_state]
        if not isfinite(value):
            raise ValueError("Accumulated path cost is outside the supported numeric range.")
        return value

    def distance(self, state, other):
        x, y = self.coordinates[state]
        other_x, other_y = self.coordinates[other]
        value = hypot(x - other_x, y - other_y)
        if not isfinite(value):
            raise ValueError("Coordinate distance is outside the supported numeric range.")
        return value

    def heuristics(self, objective="cost"):
        """Compute lower bounds to the closest goal; called during the search.

        objective='greedy': raw geometric distance for GBFS ordering.
        objective='cost': scale by min(edge_cost / geometric_length).
        objective='moves': scale by min(1 / geometric_length) = 1/max_length.
        Empty ratio sets give zero. Zero-length edges impose no restriction.
        """
        if objective not in {"greedy", "cost", "moves"}:
            raise ValueError("Unknown heuristic objective.")
        scale = 1.0
        if objective != "greedy":
            ratios = []
            for u, edges in self.graph.items():
                for v, cost in edges.items():
                    length = self.distance(u, v)
                    if length > 0:
                        ratios.append((1.0 if objective == "moves" else cost) / length)
            scale = min(ratios, default=0.0)
        return {state: scale * min(self.distance(state, goal) for goal in self.goals)
                for state in self.graph}


class Node:
    """A search record, distinct from the graph state it describes."""

    def __init__(self, state, parent=None, path_cost=0.0):
        self.state = state
        self.parent = parent
        self.path_cost = path_cost
        self.depth = 0 if parent is None else parent.depth + 1

    def child_node(self, problem, action):
        state = problem.result(self.state, action)
        cost = problem.path_cost(self.path_cost, self.state, action, state)
        return Node(state, self, cost)

    def path(self):
        route, node = [], self
        while node is not None:
            route.append(node.state)
            node = node.parent
        return list(reversed(route))


@dataclass
class SearchResult:
    goal: Optional[int]
    path: list
    nodes_created: int
    path_cost: Optional[float]
    nodes_expanded: int
    iterations: int = 1
    runtime_ms: float = 0.0
    trace: list = field(default_factory=list)

    @property
    def moves(self):
        return len(self.path) - 1 if self.goal is not None else None


def make_result(node, created, expanded, iterations=1, trace=None):
    return SearchResult(node.state if node is not None else None,
                        node.path() if node is not None else [], created,
                        node.path_cost if node is not None else None,
                        expanded, iterations, trace=trace or [])


NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
NODE_LINE = re.compile(rf"([-+]?\d+)\s*:\s*\(\s*({NUMBER})\s*,\s*({NUMBER})\s*\)")
EDGE_LINE = re.compile(rf"\(\s*([-+]?\d+)\s*,\s*([-+]?\d+)\s*\)\s*:\s*({NUMBER})")


def parse_problem(text):
    """Parse data as data: no eval or execution of input-file contents."""
    coordinates, pairs, origins, goals = {}, {}, [], []
    headers, section = [], None
    for lineno, raw in enumerate(text.lstrip('\ufeff').splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        if line in ("Nodes:", "Edges:", "Origin:", "Destinations:"):
            section = line[:-1]
            if section in headers:
                raise ValueError(f"Line {lineno}: duplicate section {section}.")
            headers.append(section)
            continue
        try:
            if section == "Nodes":
                match = NODE_LINE.fullmatch(line)
                if not match:
                    raise ValueError("Expected node_id: (x,y).")
                state = int(match[1])
                if state in coordinates:
                    raise ValueError("Duplicate node ID.")
                coordinates[state] = (float(match[2]), float(match[3]))
            elif section == "Edges":
                match = EDGE_LINE.fullmatch(line)
                if not match:
                    raise ValueError("Expected (from,to): cost.")
                pair = (int(match[1]), int(match[2]))
                if pair in pairs:
                    raise ValueError("Duplicate directed edge.")
                pairs[pair] = float(match[3])
            elif section == "Origin":
                if not re.fullmatch(r"[-+]?\d+", line):
                    raise ValueError("Expected one integer origin ID.")
                origins.append(int(line))
            elif section == "Destinations":
                if not re.fullmatch(r"[-+]?\d+(?:\s*;\s*[-+]?\d+)*\s*;?", line):
                    raise ValueError("Separate destination IDs with semicolons.")
                goals.extend(int(part) for part in line.split(';') if part.strip())
            else:
                raise ValueError("Expected a section header.")
        except ValueError as error:
            raise ValueError(f"Line {lineno}: {error}") from error
    if headers != ["Nodes", "Edges", "Origin", "Destinations"]:
        raise ValueError("Expected Nodes, Edges, Origin, Destinations in that order.")
    if len(origins) != 1:
        raise ValueError("Exactly one origin is required.")
    graph = {state: {} for state in coordinates}
    for (u, v), cost in pairs.items():
        if u not in graph or v not in graph:
            raise ValueError("An edge references an undefined node.")
        graph[u][v] = cost
    return RouteProblem(graph, origins[0], goals, coordinates)


def read_problem(filename):
    return parse_problem(Path(filename).read_text(encoding="utf-8-sig"))
