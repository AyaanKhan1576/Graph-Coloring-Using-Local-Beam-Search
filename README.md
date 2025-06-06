﻿
---

# Graph Coloring Using Local Beam Search

**Name:** Ayaan Khan  

## Overview

This repository contains the source code and report for an implementation of a graph coloring algorithm using Local Beam Search. The objective is to color the vertices of an undirected graph such that:
- No two adjacent vertices share the same color.
- The total number of distinct colors is minimized.
- The distribution of colors is balanced (i.e., similar numbers of vertices per color).

The algorithm also takes into account additional constraints including pre-assigned colors and distance constraints (vertices two hops away).

## Methodology

The solution is implemented in Python and follows these key steps:

1. **Graph Parsing:**  
   A `Parser` class reads a graph from a text file and builds an adjacency list, edge list, heuristic map, and a sorted list of vertices.

2. **State Representation:**  
   A `State` class represents a solution as a color assignment (a dictionary mapping each vertex to a color). It supports making a new state by changing the color of one vertex.

3. **Heuristic Evaluation:**  
   A `Heuristic` class evaluates each state based on:
   - **Constraint Violations:** Adjacency violations, distance constraints, and adherence to pre-assigned colors.
   - **Distinct Color Count:** Penalizes using too many colors.
   - **Color Balance:** Measures the standard deviation of the color usage, with lower values indicating a more balanced distribution.
   
   The final cost is a weighted sum of these factors.

4. **Local Beam Search:**  
   A `LocalBeamSearch` class implements the search by maintaining a beam (set) of candidate states. Successors are generated by changing one vertex’s color at a time (from 0 up to the current maximum color + 1). The algorithm iteratively selects the best candidates until a valid solution is found or a maximum number of iterations is reached.

## Requirements

- Python 3.x
- No external libraries are required beyond the Python standard library.


## Results

For example, using a test file, the output might be:

```
=== Local Beam Search - Graph Coloring Results ===
Best Score: 962.2222222222222
Color Assignment (vertex -> color):
  Vertex 0 -> Color 0
  Vertex 1 -> Color 1
  Vertex 2 -> Color 1
  Vertex 3 -> Color 0
  Vertex 4 -> Color 2
  Vertex 5 -> Color 0
  Vertex 6 -> Color 0
  Vertex 7 -> Color 1
  Vertex 8 -> Color 1
  Vertex 9 -> Color 0
  Vertex 10 -> Color 0
  Vertex 11 -> Color 1
  Vertex 12 -> Color 1
  Vertex 13 -> Color 1
  Vertex 14 -> Color 1
  Vertex 15 -> Color 2
Final Constraint Violations: 0
Distinct Colors Used: 3
Standard Deviation of Colours Used (Balance Score): 2.494438257849294
```

This shows that the algorithm achieves an optimal 3-coloring of the graph with no constraint violations.

## Conclusion

The implemented local beam search algorithm successfully finds a valid coloring that minimizes the number of colors while attempting to balance the load among them. The provided report discusses further details, observations, and potential improvements.

---
