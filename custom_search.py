
from math import hypot, ceil, inf


# ==========================================
# CUS1: Iterative Deepening DFS (Uninformed)
# ==========================================

def cus1_iddfs(graph, origin, destinations):
    """
    CUS1: Iterative Deepening Depth-First Search.
    Returns: goal, nodes_created, path
    """

    destinations = set(destinations)
    nodes_created = 0

    def depth_limited_search(node, path, limit):
        nonlocal nodes_created

        # Check if current node is a destination
        if node in destinations:
            return path

        # Stop when depth limit is reached
        if len(path) - 1 >= limit:
            return None

        # Explore neighbors in ascending order
        neighbors = sorted(graph.get(node, []))

        for neighbor, cost in neighbors:

            # Prevent cycles
            if neighbor in path:
                continue

            nodes_created += 1

            result = depth_limited_search(
                neighbor,
                path + [neighbor],
                limit
            )

            if result is not None:
                return result

        return None

    # Increase depth limit after each search
    for depth in range(len(graph)):
        nodes_created += 1

        result = depth_limited_search(
            origin,
            [origin],
            depth
        )

        if result is not None:
            return result[-1], nodes_created, result

    return None, nodes_created, []


# ==========================================
# CUS2: Iterative Deepening A* (Informed)
# ==========================================

def cus2_idastar(graph, coordinates, origin, destinations):
    """
    CUS2: Iterative Deepening A* Search.
    Finds a destination using the fewest moves.
    Returns: goal, nodes_created, path
    """

    goals = set(destinations)

    if not goals:
        return None, 0, []

    # Find longest straight-line edge distance
    max_edge_distance = max(
        (
            hypot(
                coordinates[u][0] - coordinates[v][0],
                coordinates[u][1] - coordinates[v][1]
            )
            for u, edges in graph.items()
            for v, cost in edges
        ),
        default=0
    )

    # Heuristic: estimated minimum remaining moves
    def heuristic(node):
        if max_edge_distance == 0:
            return 0

        x, y = coordinates[node]

        closest = min(
            hypot(
                x - coordinates[goal][0],
                y - coordinates[goal][1]
            )
            for goal in goals
        )

        return max(
            0,
            ceil(closest / max_edge_distance - 1e-12)
        )

    bound = heuristic(origin)
    created = 0

    def search(path, limit):
        nonlocal created

        node = path[-1]
        moves = len(path) - 1
        score = moves + heuristic(node)

        # Stop exploring if score exceeds limit
        if score > limit:
            return None, score

        # Check if destination is reached
        if node in goals:
            return path, None

        next_limit = inf

        # Expand neighbors in ascending order
        for neighbor, cost in sorted(graph.get(node, [])):

            # Prevent cycles
            if neighbor in path:
                continue

            created += 1

            found, exceeded = search(
                path + [neighbor],
                limit
            )

            if found is not None:
                return found, None

            if exceeded is not None:
                next_limit = min(next_limit, exceeded)

        return None, next_limit

    # Iteratively increase the search threshold
    while True:
        created += 1

        path, new_limit = search([origin], bound)

        if path is not None:
            return path[-1], created, path

        if new_limit == inf:
            return None, created, []

        bound = new_limit


# ==========================================
# MAIN PROGRAM - TESTING BOTH ALGORITHMS
# ==========================================

if __name__ == "__main__":

    # Graph connections and edge costs
    graph = {
        1: [(3, 5), (4, 6)],
        2: [(1, 4), (3, 4)],
        3: [(1, 5), (2, 5), (5, 6), (6, 7)],
        4: [(1, 6), (3, 5), (5, 7)],
        5: [(3, 6), (4, 8)],
        6: [(3, 7)]
    }

    # Coordinates for each graph node
    coordinates = {
        1: (4, 1),
        2: (2, 2),
        3: (4, 4),
        4: (6, 3),
        5: (5, 6),
        6: (7, 5)
    }

    # Origin and destination nodes
    origin = 2
    destinations = {4, 5}

    # ======================================
    # TEST CUS1
    # ======================================

    goal, created, path = cus1_iddfs(
        graph, origin, destinations
    )

    print("Algorithm: CUS1 (IDDFS)")
    print("Goal:", goal)
    print("Nodes created:", created)
    print("Path:", " -> ".join(map(str, path)))

    # ======================================
    # TEST CUS2
    # ======================================

    goal, created, path = cus2_idastar(
        graph, coordinates, origin, destinations
    )

    print("\nAlgorithm: CUS2 (IDA*)")
    print("Goal:", goal)
    print("Nodes created:", created)
    print("Path:", " -> ".join(map(str, path)))
