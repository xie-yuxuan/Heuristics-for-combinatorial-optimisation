# Heuristics-for-combinatorial-optimisation

This is my IIB project at the University of Cambridge. My specialisation is in Information and Computer Engineering. 

### Objectives 
- Explore reluctant vs greedy algo for graph coloring optimisation
- Explore heuristics of approaches

### To do 
1. Graph representation (done)
2. Greedy and Reluctant algo (done)
3. Optimisation iteration 
4. Validation and visualisation

### Methods to reduce complexity
- Calculating change (delta) in cost for a recoloring rather than calculating the whole cost of the graph for each iteration
- Maintaining a priority queue (sorted list) of vertices based on possible cost reduction, queue is updated aft each iteration to ensure vertex with max potential in cost reduction is evaluated first
- naive greedy complexity if O(n x k) where n is the number of nodes and k is the color set size
- incorporate heuristic methods like simulated annealing or genetic algo to efficiently explore search space

### Future work
- Explore random graphs 
- Explore mix of algos
- Draw edge weights from probability distributions
- Allow negative edge weights
