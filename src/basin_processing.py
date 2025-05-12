import json
import os
import numpy as np
import networkx as nx
import time
import copy
from networkx.readwrite import json_graph

from visualisation import draw_graph
from algorithms import optimise, optimise2, optimise3, optimise4, optimise_random
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

def fix_reflected_entries_in_json(file_path, num_nodes):
    with open(file_path, 'r') as f:
        data = json.load(f)

    max_key = 2 ** num_nodes - 1

    for key in ["Sgr1", "Srr1"]:
        basin = data["basin_data"].get(key, {})
        new_entries = {}

        for k_str, v in basin.items():
            k = int(k_str)
            reflected_k = max_key - k
            reflected_v = max_key - v
            if str(reflected_k) not in basin:
                new_entries[str(reflected_k)] = reflected_v

        basin.update(new_entries)
        data["basin_data"][key] = basin

    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Fixed reflected entries for Sgr1 and Srr1 in {file_path}")


if __name__ == "__main__":

    seed = 1
    np.random.seed(seed)


    file_path = r"C:\Projects\Heuristics for combinatorial optimisation\Heuristics-for-combinatorial-optimisation\data\graphs\(14, 13, 2, 5).json"
    graph, graph_name, color_set_size, degree, num_nodes, gaussian_mean, gaussian_variance, initial_node_colors = load_graph_from_json(file_path)
    
    graphs_path = r"C:\Projects\Heuristics for combinatorial optimisation\results"
    os.makedirs(graphs_path, exist_ok=True)
    output_file = os.path.join(graphs_path, f"{graph_name}_basin_results.json")

    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            results = json.load(f)
    else:
        results = {
            "graph_name": graph_name,
            "degree": degree,
            "num_nodes": num_nodes,
            "color_set_size": color_set_size,
            "gaussian_mean": gaussian_mean,
            "gaussian_variance": gaussian_variance,
            "basin_data": {}
        }

    Sg = {}
    Sr = {}
    # Sgr1 = {}
    # Srr1 = {}
    # Sgr3 = {}
    # Srr3 = {}

    for i in range(2 ** (num_nodes -1)):
    # for i in range(1):
        binary_coloring = format(i, f'0{num_nodes}b')

        # Assign colors to nodes for original coloring
        for node_idx, color in enumerate(binary_coloring):
            graph.nodes[node_idx]['color'] = int(color)

        # graph_copy1 = copy.deepcopy(graph)
        # graph_copy2 = copy.deepcopy(graph)
        # graph_copy3 = copy.deepcopy(graph)
        graph_copy4 = copy.deepcopy(graph)
        # graph_copy5 = copy.deepcopy(graph)
        # graph_copy6 = copy.deepcopy(graph)

        graph_g, final_cost_g, iterations_taken_g, cost_data_g = optimise4(graph, color_set_size, algo_func=fg)
        # graph_gr1, final_cost_gr1, iterations_taken_gr1, cost_data_gr1 = optimise_random(graph_copy2, color_set_size, algo_func=fg, random_prob=0.1)
        # graph_gr3, final_cost_gr3, iterations_taken_gr3, cost_data_gr3 = optimise_random(graph_copy3, color_set_size, algo_func=fg, random_prob=0.3)
        # print(final_cost_g)
        # draw_graph(graph_g, pos=nx.spring_layout(graph, seed=1), graph_name=None, iterations_taken=None, cost_data=None, 
        #     color_set_size=2, 
        #     degree=5, 
        #     num_nodes=10, 
        #     gaussian_mean=0, 
        #     gaussian_variance=1, 
        #     ground_truth_log_likelihood=None)

        graph_r, final_cost_r, iterations_taken_r, cost_data_r = optimise4(graph_copy4, color_set_size, algo_func=fr)
        # graph_rr1, final_cost_rr1, iterations_taken_rr1, cost_data_rr1 = optimise_random(graph_copy5, color_set_size, algo_func=fr, random_prob=0.1)
        # graph_rr3, final_cost_rr3, iterations_taken_rr3, cost_data_rr3 = optimise_random(graph_copy6, color_set_size, algo_func=fr, random_prob=0.3)

        final_coloring_g = int("".join(str(graph_g.nodes[node]['color']) for node in range(graph_g.number_of_nodes())), 2)
        final_coloring_r = int("".join(str(graph_r.nodes[node]['color']) for node in range(graph_r.number_of_nodes())), 2)
        # final_coloring_gr1 = int("".join(str(graph_gr1.nodes[node]['color']) for node in range(graph_gr1.number_of_nodes())), 2)
        # final_coloring_rr1 = int("".join(str(graph_rr1.nodes[node]['color']) for node in range(graph_rr1.number_of_nodes())), 2)
        # final_coloring_gr3 = int("".join(str(graph_gr3.nodes[node]['color']) for node in range(graph_gr3.number_of_nodes())), 2)
        # final_coloring_rr3 = int("".join(str(graph_rr3.nodes[node]['color']) for node in range(graph_rr3.number_of_nodes())), 2)

        # Store original results
        Sg[i] = final_coloring_g
        Sr[i] = final_coloring_r
        # Sgr1[i] = final_coloring_gr1
        # Srr1[i] = final_coloring_rr1
        # Sgr3[i] = final_coloring_gr3
        # Srr3[i] = final_coloring_rr3

        # # Reflect results for the other half
        Sg[2 ** num_nodes - 1 - i] = 2 ** num_nodes - 1 - final_coloring_g
        Sr[2 ** num_nodes - 1 - i] = 2 ** num_nodes - 1 - final_coloring_r
        # Sgr1[2 ** num_nodes - 1 - i] = 2 ** num_nodes - 1 - final_coloring_gr1
        # Srr1[2 ** num_nodes - 1 - i] = 2 ** num_nodes - 1 - final_coloring_rr1
        # Sgr3[2 ** num_nodes - 1 - i] = 2 ** num_nodes - 1 - final_coloring_gr3
        # Srr3[2 ** num_nodes - 1 - i] = 2 ** num_nodes - 1 - final_coloring_rr3

        if i % 1000 == 0:
            print(f"{i} initial coloring optimisation complete")

    results["basin_data"]["Sg"] = Sg
    results["basin_data"]["Sr"] = Sr
    # results["basin_data"]["Sgr1"] = Sgr1
    # results["basin_data"]["Srr1"] = Srr1
    # results["basin_data"]["Sgr3"] = Sgr3
    # results["basin_data"]["Srr3"] = Srr3

    graphs_path = r"C:\Projects\Heuristics for combinatorial optimisation\results"
    os.makedirs(graphs_path, exist_ok=True)

    output_file = os.path.join(graphs_path, f"{graph_name}_basin_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Saved results to {output_file}")