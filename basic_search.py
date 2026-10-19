"""Uninformed search algorithms for COS30019 Assignment 2.

This file implements:
- Breadth-First Search (BFS)
- Depth-First Search (DFS)
"""

from collections import deque
from route_problem import Node, make_result

def breadth_first_search(problem, trace=False):
    """Search the graph level by level using Breadth-First Search."""

    # Create the starting node
    root = Node(problem.initial)

    # BFS uses a FIFO queue
    frontier = deque([root])

    # Keep track of states already discovered
    discovered = {root.state}

    # Assignment asks for number of nodes created
    created = 1

    # Keep track of expanded nodes
    expanded = 0

    history = []

    while frontier:

        # Remove the oldest node from the queue
        node = frontier.popleft()

        # Check whether this node is a destination
        if problem.goal_test(node.state):
            return make_result(
                node,
                created,
                expanded,
                trace=history
            )

        expanded += 1

        # problem.actions() already returns neighbours
        # in ascending node order
        for action in problem.actions(node.state):

            next_state = problem.result(
                node.state,
                action
            )

            # Do not create the same state repeatedly
            if next_state not in discovered:

                discovered.add(next_state)

                child = node.child_node(
                    problem,
                    action
                )

                created += 1

                frontier.append(child)

    # No destination could be reached
    return make_result(
        None,
        created,
        expanded,
        trace=history
    )

def depth_first_search(problem, trace=False):
    """Search the graph by exploring one branch as deeply as possible."""

    # Create the starting node
    root = Node(problem.initial)

    # DFS uses a stack (LIFO)
    frontier = [root]

    # Keep track of discovered states
    discovered = {root.state}

    # Number of nodes created
    created = 1

    # Number of nodes expanded
    expanded = 0

    history = []

    while frontier:

        # Remove the newest node from the stack
        node = frontier.pop()

        # Check whether this node is a destination
        if problem.goal_test(node.state):
            return make_result(
                node,
                created,
                expanded,
                trace=history
            )

        expanded += 1

        # Reverse because stack is LIFO.
        # This ensures smaller-numbered nodes
        # are expanded first.
        actions = problem.actions(node.state)

        for action in reversed(actions):

            next_state = problem.result(
                node.state,
                action
            )

            if next_state not in discovered:

                discovered.add(next_state)

                child = node.child_node(
                    problem,
                    action
                )

                created += 1

                frontier.append(child)

    # No destination can be reached
    return make_result(
        None,
        created,
        expanded,
        trace=history
    )