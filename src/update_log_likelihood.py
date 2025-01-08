import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os

from utils import calc_log_likelihood
from sbm_graph_processing import load_graph_from_json





if __name__ == '__main__':
    graphs_path = r"C:\Projects\Heuristics for combinatorial optimisation\Heuristics-for-combinatorial-optimisation\data\graphs\SBM(40, 4, c).json"  # Use raw string for Windows path
    graph, graph_name, num_nodes, num_groups, group_mode, initial_node_colors, ground_truth_log_likelihood = load_graph_from_json(graphs_path)

    # Open the file and load the data
    with open(graphs_path, 'r') as f:
        data = json.load(f)

    # Generate the w matrix (edge probabilities)
    w = np.zeros((num_groups, num_groups))

    if group_mode == "association":
        w += 0.1  # Small baseline for non-diagonal elements
        np.fill_diagonal(w, 0.9)  # Large diagonal elements
    elif group_mode == "bipartite":
        w += 0.9  # Large baseline for non-diagonal elements
        np.fill_diagonal(w, 0.1)  # Small diagonal elements
    elif group_mode == "core-periphery":
        w += 0.9  # Large baseline
        w[0, :] = 0.1  # Small first row (loners have low connections to all groups)
        w[:, 0] = 0.1  # Small first column (low connections to loners)
        w[0, 0] = 0.1  # loners have low self-connections

    print(calc_log_likelihood(graph, w))
    # Update the value in the data dictionary
    data['ground_truth_log_likelihood'] = calc_log_likelihood(graph, w)

    # Write the updated data back to the file
    with open(graphs_path, 'w') as f:
        json.dump(data, f, indent=2)  # Use json.dump to write the data with formatting