"""Stage 3 independent tests for Member 3. Place next to custom_search.py."""
from collections import deque
from pathlib import Path

from custom_search import cus1_iddfs, cus2_idastar

CASES = {
    "test_11_large_graph.txt": 5,
    "test_12_long_route.txt": 10,
    "test_13_many_branches.txt": 2,
    "test_14_merging_paths.txt": 3,
    "test_15_difficult_combo.txt": 2,
}


def read_graph(filepath):
    """Simple temporary parser; Member 1's parser can replace this later."""
    lines = [line.strip() for line in Path(filepath).read_text(encoding="utf-8").splitlines() if line.strip()]
    section = None
    coordinates = {}
    graph = {}
    origin = None
    goals = set()
    for line in lines:
        if line in ("Nodes:", "Edges:", "Origin:", "Destinations:"):
            section = line[:-1]
            continue
        if section == "Nodes":
            number, position = line.split(":", 1)
            x, y = position.strip().strip("()").split(",")
            node = int(number)
            coordinates[node] = (float(x), float(y))
            graph[node] = []
        elif section == "Edges":
            pair, weight = line.split(":", 1)
            a, b = pair.strip().strip("()").split(",")
            graph[int(a)].append((int(b), float(weight)))
        elif section == "Origin":
            origin = int(line)
        elif section == "Destinations":
            goals = {int(item.strip()) for item in line.split(";")}
    return graph, coordinates, origin, goals


def check_result(graph, origin, goals, goal, path, expected_moves):
    """Check goal validity, directed connectivity, and optimal number of moves."""
    if not path or path[0] != origin or path[-1] != goal or goal not in goals:
        return False
    for a, b in zip(path, path[1:]):
        if not any(next_node == b for next_node, weight in graph.get(a, [])):
            return False
    return len(path) - 1 == expected_moves


def route_cost(graph, path):
    return sum(next(w for node, w in graph[a] if node == b) for a, b in zip(path, path[1:]))


if __name__ == "__main__":
    folder = Path(__file__).parent
    passed = 0
    for filename, expected_moves in CASES.items():
        graph, coords, origin, goals = read_graph(folder / filename)
        print(f"\n=== {filename} | shortest route: {expected_moves} moves ===")
        for method, algorithm in (
            ("CUS1", lambda: cus1_iddfs(graph, origin, goals)),
            ("CUS2", lambda: cus2_idastar(graph, coords, origin, goals)),
        ):
            goal, created, path = algorithm()
            ok = check_result(graph, origin, goals, goal, path, expected_moves)
            passed += int(ok)
            print(f"{method}: {'PASS' if ok else 'FAIL'} | goal={goal} | "
                  f"moves={len(path)-1 if path else 'n/a'} | "
                  f"cost={route_cost(graph, path) if path else 'n/a'} | "
                  f"created={created} | path={path}")
    print(f"\nResults: {passed}/{2 * len(CASES)} checks passed")
