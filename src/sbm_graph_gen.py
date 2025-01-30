import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
from networkx.readwrite import json_graph

from visualisation import draw_graph
from utils import calc_log_likelihood, compute_w

def gen_sbm_graph(g, w):
    num_nodes = len(g)
    num_groups = w.shape[0]

    group_membership = {i: int(g[i]) for i in range(num_nodes)}
    
    # Generate the adjacency matrix
    adjacency_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            prob = w[group_membership[i], group_membership[j]]
            if np.random.rand() < prob:
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1

    # Create the NetworkX graph
    graph = nx.from_numpy_array(adjacency_matrix)

    # Assign colors to nodes based on group membership
    for node in graph.nodes():
        group = group_membership[node]
        graph.nodes[node]['color'] = group

    return graph, adjacency_matrix

if __name__ == '__main__':
    # Set a random seed for reproducibility
    seed = 1
    np.random.seed(seed)

    # set parameters
    num_nodes = 100
    num_groups = 2
    num_initial_colorings = 100
    group_mode = ["association", "bipartite", "core-periphery"][0]
    graph_name = f"SBM({num_nodes}, {num_groups}, {group_mode[0]})"

    # Generate the g vector (color assignment)
    g = []
    for group in range(num_groups):
        g.extend([group] * (num_nodes // num_groups))
    g.extend([num_groups - 1] * (num_nodes % num_groups))
    g = np.array(g)

    # Generate the w matrix (edge probabilities)
    w = np.zeros((num_groups, num_groups))

    if group_mode == "association":
        w += 1  # Small baseline for non-diagonal elements
        np.fill_diagonal(w, 9)  # Large diagonal elements
    elif group_mode == "bipartite":
        w += 9  # Large baseline for non-diagonal elements
        np.fill_diagonal(w, 1)  # Small diagonal elements
    elif group_mode == "core-periphery":
        w += 9  # Large baseline
        w[0, :] = 1  # Small first row (loners have low connections to all groups)
        w[:, 0] = 1  # Small first column (low connections to loners)
        w[0, 0] = 1  # loners have low self-connections

    w /= num_nodes

    graph, adjacency_matrix = gen_sbm_graph(g, w)

    # uncomment to view graphs before saving
    draw_graph(graph, pos=nx.spring_layout(graph, seed=1), graph_name=graph_name, iterations_taken=0, cost_data=None,
               color_set_size=num_groups, 
               degree=None, 
               num_nodes=num_nodes, 
               gaussian_mean=None, 
               gaussian_variance=None,
               ground_truth_log_likelihood = None
               )
    
    # print("Color assignment vector (g):")
    # print(g)
    # print("Edge probability matrix (w):")
    # print(w)
    # print(adjacency_matrix)

    # save graph into graphs folder
    graphs_path = "C:\Projects\Heuristics for combinatorial optimisation\Heuristics-for-combinatorial-optimisation\data\graphs"

    graph_data = json_graph.node_link_data(graph) # node_link_data converts graph into dictionary to be serialised to JSON
    
    # create a list of initial color states (list of lists)
    initial_node_colors = [
        [np.random.randint(0, num_groups) for _ in range(num_nodes)]
        for _ in range(num_initial_colorings)
    ]


    n, m = np.zeros(num_groups), np.zeros((num_groups, num_groups))

    for node in graph.nodes():
        n[g[node]] += 1 # increment group count for each group

    for u, v in graph.edges():
        # increment edge count between groups
        # ensures m is symmetric
        m[g[v], g[u]] = m[g[u], g[v]] = m[g[u], g[v]] + 1


    data = {
        "graph_name": graph_name,
        "num_nodes" : num_nodes,
        "num_groups" : num_groups,
        "group_mode" : group_mode,
        "graph_data": graph_data,
        "ground_truth_log_likelihood": calc_log_likelihood(n, m, w), #TODO: ground truth maybe not this, but instead the educated guess of w
        "initial_node_colors": initial_node_colors
    }

    with open(os.path.join(graphs_path, f"{graph_name}.json"), 'w') as f:
        json.dump(data, f, indent = 2)

    print(f"Saved graph to {graphs_path}/{graph_name}.json")