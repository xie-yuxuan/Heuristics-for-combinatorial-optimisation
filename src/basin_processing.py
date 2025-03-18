import json
import os
import numpy as np
import networkx as nx
import time
import copy
from networkx.readwrite import json_graph

from visualisation import draw_graph
from algorithms import optimise, optimise2, optimise3, optimise4
from graph_gen import generate_random_regular_graph

def load_graph_from_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # load graph_name and attibutes
    graph_name = data["graph_name"]
    color_set_size = data["color_set_size"]
    degree = data["degree"]
    num_nodes = data["num_nodes"]
    gaussian_mean = data["gaussian_mean"]
    gaussian_variance = data["gaussian_variance"]
    initial_node_colors = data["initial_node_colors"]
    
    graph_data = data["graph_data"]
    graph = json_graph.node_link_graph(graph_data)

    # uncomment to get adj matrix
    # adj_matrix = nx.adjacency_matrix(graph, weight='weight').toarray()
    
    return graph, graph_name, color_set_size, degree, num_nodes, gaussian_mean, gaussian_variance, initial_node_colors

fg = lambda x: x # greedy transforamtion to cost change matrix
def fr(x): # reluctant transformation to cost change matrix
    # check if x is a np array
    if isinstance(x, np.ndarray):
        vectorized_func = np.vectorize(lambda x: 0.0 if x == 0 else 1.0 / x) # vectorize to handle each ele individually
        return vectorized_func(x)
    else:
        return 0.0 if x == 0 else 1.0 / x

if __name__ == "__main__":
    file_path = r"C:\Projects\Heuristics for combinatorial optimisation\Heuristics-for-combinatorial-optimisation\data\graphs\(16, 8, 2).json"
    graph, graph_name, color_set_size, degree, num_nodes, gaussian_mean, gaussian_variance, initial_node_colors = load_graph_from_json(file_path)
    
    results = {
        "graph_name": graph_name,
        "degree" : degree,
        "num_nodes" : num_nodes,
        "color_set_size" : color_set_size,
        "gaussian_mean" : gaussian_mean,
        "gaussian_variance" : gaussian_variance,
        "basin_data" : {}
    }

    Sg = {}
    Sr = {}

    # Enumerate all possible colorings (2^num_nodes)
    for i in range(2 ** num_nodes):
    # for i in range(4):
        # Convert number to binary string representing coloring
        binary_coloring = format(i, f'0{num_nodes}b')  # Ensures fixed-length binary

        # Assign colors to nodes
        for node_idx, color in enumerate(binary_coloring):
            graph.nodes[node_idx]['color'] = int(color)

        # Deep copy for separate optimizations
        graph_copy1 = copy.deepcopy(graph)

        # Run optimizations
        graph_g, final_cost_g, iterations_taken_g, cost_data_g = optimise4(graph, color_set_size, algo_func=fg)
        graph_r, final_cost_r, iterations_taken_r, cost_data_r = optimise4(graph_copy1, color_set_size, algo_func=fr)

        # Convert final coloring to binary representation (integer form)
        final_coloring_g = int("".join(str(graph_g.nodes[node]['color']) for node in graph_g.nodes), 2)
        final_coloring_r = int("".join(str(graph_r.nodes[node]['color']) for node in graph_r.nodes), 2)

        # Store results
        Sg[i] = final_coloring_g
        Sr[i] = final_coloring_r
        
        if i % 1000 == 0:
            print(f"{i} initial coloring optimisation complete")

    # Add basin data to results
    results["basin_data"] = {"Sg": Sg, "Sr": Sr}

    # Save results as JSON
    graphs_path = r"C:\Projects\Heuristics for combinatorial optimisation\results"
    os.makedirs(graphs_path, exist_ok=True)

    output_file = os.path.join(graphs_path, f"{graph_name}_basin_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Saved results to {output_file}")