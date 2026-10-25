# COS30019 – Introduction to Artificial Intelligence
## Assignment 2 Part A: Tree-Based Search (Route Finding)

A Python command-line application that solves directed, weighted route-finding problems using six search algorithms. The program reads a graph from a `.txt` file, searches for **any one** of its destination nodes, and prints the goal, number of search nodes created, and solution path.

## Team Members

| Name | Student ID |
| --- | --- |
| Darren Chong Khai Chen | 104406534 |
| Shiaw Chwen Heng | 102781527 |
| Benson Kiing Xie Yew | 104405845 |

## Implemented Algorithms

| Command | Algorithm | Type | Search objective |
| --- | --- | --- | --- |
| `DFS` | Depth-First Search | Uninformed | Find a valid path |
| `BFS` | Breadth-First Search | Uninformed | Find a path with the fewest moves |
| `GBFS` | Greedy Best-First Search | Informed | Use heuristic guidance to find a path |
| `AS` | A* Search | Informed | Find a path with minimum weighted cost |
| `CUS1` | Iterative Deepening Depth-First Search (IDDFS) | Custom, uninformed | Find a path with the fewest moves |
| `CUS2` | Iterative Deepening A* (IDA*) | Custom, informed | Find a path with the fewest moves using a heuristic |

## Requirements

- Python 3
- Windows Command Prompt or the integrated terminal in VS Code
- No third-party Python packages are required for the command-line search program.

## How to Run

1. Open the project folder in VS Code.
2. Open **Terminal → New Terminal**.
3. Run the program using this format:

   ```bash
   python search.py <filename> <method>
   ```

For example, to run both custom algorithms using the supplied sample graph:

```bash
python search.py test_case/PathFinder-test-1.txt CUS1
python search.py test_case/PathFinder-test-1.txt CUS2
```

Expected results:

**CUS1**

```text
test_case/PathFinder-test-1.txt CUS1
4 8
2; 1; 4
```

**CUS2**

```text
test_case/PathFinder-test-1.txt CUS2
4 4
2; 1; 4
```

The first line shows the input filename and selected method. The second line shows the reached goal and the number of search nodes created. The third line shows the route from origin to destination.

## Input File Format

Graph inputs use four sections:

```text
Nodes:
1: (0,0)
2: (1,0)
3: (2,0)
Edges:
(1,2): 4
(2,3): 5
Origin:
1
Destinations:
3
```

Edges are **directed**: `(1,2): 4` permits travel from node 1 to node 2, but not necessarily from node 2 to node 1. Multiple destinations can be separated by semicolons, such as `3; 5`.

## Project Files

| File or folder | Purpose |
| --- | --- |
| `search.py` | Command-line entry point and algorithm selection |
| `route_problem.py` | Input-file parser, graph representation, search nodes, and result objects |
| `basic_search.py` | DFS and BFS implementations |
| `inform_search.py` | GBFS and A* implementations |
| `custom_search.py` | CUS1 (IDDFS) and CUS2 (IDA*) implementations |
| `test_case/` | Assignment sample and all 16 numbered graph test cases (`testcase1.txt`–`testcase16.txt`) |
| `test_member3.py` | Automated CUS1/CUS2 checks on `testcase12.txt`–`testcase16.txt` |
| `verify_project.py` | Runs all six algorithms on the 17 graph files and two built-in edge cases |

## Testing

The project includes **17 graph input files**: `PathFinder-test-1.txt` and `testcase1.txt` through `testcase16.txt`. The test scenarios cover directed connections, cycles, branching, multiple destinations, unreachable goals, varying edge weights, and larger graphs.

To run the custom-search checks on `testcase12.txt` through `testcase16.txt`:

```bash
python test_member3.py
```

Expected summary:

```text
Results: 10/10 checks passed
```

These checks validate the paths returned by CUS1 and CUS2 against independently calculated minimum move counts. To run all six algorithms against all 17 graph files plus two built-in edge cases:

```bash
python verify_project.py
```

`test_custom_integration.py` is retained with its original file references; use `test_member3.py` for the renamed Stage 3 test cases. The report should present the checked results from the full test set.

## Notes

- A shorter route in **moves** is not always the route with the lowest **weighted cost**.
- The required command-line output should not include extra debug messages.
- Report and submission documentation should acknowledge any AI assistance used, as required by the assignment.
