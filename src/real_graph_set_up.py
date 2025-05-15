import pandas as pd
import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
from networkx.readwrite import json_graph
from collections import defaultdict

from visualisation import draw_graph
from utils import calc_log_likelihood, compute_w

def convert_numpy(obj):
    if isinstance(obj, dict):
        return {convert_numpy(k): convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(x) for x in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

if __name__ == '__main__':

    np.random.seed(seed=1)

    graph_name = "wikipedia_map_of_science"  # Set graph name
    num_groups = 2  # Set number of groups
    group_modes = ["association", "bipartite", "core-periphery"]
    group_mode = group_modes[0]  # Choose one of the modes

    base_path = r'C:\Projects\Heuristics for combinatorial optimisation\Heuristics-for-combinatorial-optimisation\data\real_world_data'
    real_world_data_path = os.path.join(base_path, graph_name)

    edges_path = os.path.join(real_world_data_path, 'edges.csv')
    nodes_path = os.path.join(real_world_data_path, 'nodes.csv')

    # construct network -------------------------------------------------------
    edges_df = pd.read_csv(edges_path)
    nodes_df = pd.read_csv(nodes_path)

    nodes_df.columns = nodes_df.columns.str.strip()
    edges_df.columns = edges_df.columns.str.strip()

    # Initialize graph
    graph = nx.Graph()

    # Add nodes with attributes
    for _, row in nodes_df.iterrows():
        node_id = row['# index']
        graph.add_node(node_id)

    # Add edges with attributes
    for _, row in edges_df.iterrows():
        # if row['weight'] > 0:
        source = int(row['# source'])
        target = int(row['target'])
        weight = 1  # You can use other attributes like fiber_length_mean, etc.
        graph.add_edge(source, target, weight=weight)

    num_nodes = int(len(graph.nodes))

    # Plot the graph
    # draw_graph(graph, pos=nx.spring_layout(graph, seed=1), graph_name=graph_name, iterations_taken=None, cost_data=None, 
    #            color_set_size=None, 
    #            degree=None, 
    #            num_nodes=num_nodes,
    #            gaussian_mean=None, 
    #            gaussian_variance=None, 
    #            ground_truth_log_likelihood=None)
    
    g = []
    for group in range(num_groups):
        g.extend([group] * (num_nodes // num_groups))
    g.extend([num_groups - 1] * (num_nodes % num_groups))
    g = np.array(g)

    g_copy = g.copy()
    np.random.shuffle(g_copy)

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

    # save graph data
    graphs_path = "C:\Projects\Heuristics for combinatorial optimisation\Heuristics-for-combinatorial-optimisation\data\graphs"

    graph_data = json_graph.node_link_data(graph)
    
    initial_node_colors = [
        [np.random.randint(0, num_groups) for _ in range(num_nodes)]
        for _ in range(100)
    ]

    n, m = np.zeros(num_groups), np.zeros((num_groups, num_groups))
    for node in graph.nodes():
        n[g_copy[node]] += 1
    for u, v in graph.edges():
        m[g_copy[v], g_copy[u]] = m[g_copy[u], g_copy[v]] = m[g_copy[u], g_copy[v]] + 1

    w_json = json.dumps(w.tolist())

    data = {
        "graph_name": graph_name,
        "num_nodes": num_nodes,
        "num_groups": num_groups,
        "group_mode": group_mode,
        "graph_data": graph_data,
        "ground_truth_w": w_json,
        "ground_truth_log_likelihood": calc_log_likelihood(n, m, w),
        "initial_node_colors": initial_node_colors
    }

    # print(data)

    with open(os.path.join(graphs_path, f"{graph_name}, {group_mode[0]}.json"), 'w') as f:
        json.dump(convert_numpy(data), f, indent=2)

    print(f"Saved graph to {graphs_path}/{graph_name}, {group_mode[0]}.json")





