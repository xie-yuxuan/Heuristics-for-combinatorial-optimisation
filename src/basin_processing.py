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
    file_path = r"C:\Projects\Heuristics for combinatorial optimisation\Heuristics-for-combinatorial-optimisation\data\graphs\(20, 10, 2).json"
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

    for i in range(2 ** (num_nodes -1)):
    # for i in range(1):
        binary_coloring = format(i, f'0{num_nodes}b')

        # Assign colors to nodes for original coloring
        for node_idx, color in enumerate(binary_coloring):
            graph.nodes[node_idx]['color'] = int(color)

        graph_copy1 = copy.deepcopy(graph)

        graph_g, final_cost_g, iterations_taken_g, cost_data_g = optimise4(graph, color_set_size, algo_func=fg)
        # print(final_cost_g)
        # draw_graph(graph_g, pos=nx.spring_layout(graph, seed=1), graph_name=None, iterations_taken=None, cost_data=None, 
        #     color_set_size=2, 
        #     degree=5, 
        #     num_nodes=10, 
        #     gaussian_mean=0, 
        #     gaussian_variance=1, 
        #     ground_truth_log_likelihood=None)

        graph_r, final_cost_r, iterations_taken_r, cost_data_r = optimise4(graph_copy1, color_set_size, algo_func=fr)

        final_coloring_g = int("".join(str(graph_g.nodes[node]['color']) for node in range(graph_g.number_of_nodes())), 2)
        final_coloring_r = int("".join(str(graph_r.nodes[node]['color']) for node in range(graph_r.number_of_nodes())), 2)

        # Store original results
        Sg[i] = final_coloring_g
        Sr[i] = final_coloring_r

        # # Reflect results for the other half
        Sg[2 ** num_nodes - 1 - i] = 2 ** num_nodes - 1 - final_coloring_g
        Sr[2 ** num_nodes - 1 - i] = 2 ** num_nodes - 1 - final_coloring_r

        if i % 1000 == 0:
            print(f"{i} initial coloring optimisation complete")

    results["basin_data"] = {"Sg": Sg, "Sr": Sr}

    graphs_path = r"C:\Projects\Heuristics for combinatorial optimisation\results"
    os.makedirs(graphs_path, exist_ok=True)

    output_file = os.path.join(graphs_path, f"{graph_name}_basin_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Saved results to {output_file}")