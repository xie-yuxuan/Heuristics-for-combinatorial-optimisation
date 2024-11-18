# Heuristics-for-combinatorial-optimisation

This is my IIB project at the University of Cambridge. My specialisation is in Information and Computer Engineering. 

### Objectives 
- Explore reluctant vs greedy algo for graph coloring optimisation
- Explore heuristics of approaches

### Running the code to get results
1. graph_gen.py to generate graph J and list of initial colorings. save graph data and initial colorings to JSON
2. graph_processing.py to run greedy and reluctant optimisation algo for all initial colorings, save results of cost data to JSON
3. analysis.py to draw conclusions and plots from results e.g. cost against iterations, final cost against initial coloring index 

### Note
- Full cost data for all iterations only available for (x,20,2) and (x,20,8) to reduce file size