import json
import os
import numpy as np
import networkx as nx
import time
import copy
import seaborn as sns
from networkx.readwrite import json_graph
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix

import matplotlib.pyplot as plt

from algorithms import optimise_sbm, optimise_sbm2, optimise_sbm3, optimise_sbm4, optimise_sbm5, SBMState
from visualisation import draw_graph, generate_heatmap_of_color_changes



def load_graph_from_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # load graph_name and attibutes
    graph_name = data["graph_name"]
    num_groups = data["num_groups"]
    num_nodes = data["num_nodes"]
    group_mode = data["group_mode"]
    initial_node_colors = data["initial_node_colors"]
    ground_truth_w = data["ground_truth_w"]
    ground_truth_log_likelihood = data["ground_truth_log_likelihood"]

    graph_data = data["graph_data"]
    graph = json_graph.node_link_graph(graph_data)

    # uncomment to get adj matrix
    # adj_matrix = nx.adjacency_matrix(graph, weight='weight').toarray()
    
    return graph, graph_name, num_nodes, num_groups, group_mode, initial_node_colors, ground_truth_w, ground_truth_log_likelihood

# matrix transformation fn, depending on algo
fg = lambda x: x # greedy transforamtion to cost change matrix
def fr(x): # reluctant transformation to cost change matrix
    # check if x is a np array
    if isinstance(x, np.ndarray):
        vectorized_func = np.vectorize(lambda x: 0.0 if x == 0 else 1.0 / x) # vectorize to handle each ele individually
        return vectorized_func(x)
    else:
        return 0.0 if x == 0 else 1.0 / x

if __name__ == "__main__":
    file_path = r"C:\Projects\Heuristics for combinatorial optimisation\Heuristics-for-combinatorial-optimisation\data\graphs\SBM(1000, 2, a).json"
    graph, graph_name, num_nodes, num_groups, group_mode, initial_node_colors, ground_truth_w, ground_truth_log_likelihood = load_graph_from_json(file_path)

    color_map_path = 'C:/Projects/Heuristics for combinatorial optimisation/Heuristics-for-combinatorial-optimisation/data/color_map.json'
    with open(color_map_path, 'r') as f:
        color_map = json.load(f)['color_map']
    color_to_int = {name: int(key) for key, name in color_map.items()}

    results = {
        "graph_name": graph_name,
        "num_nodes" : num_nodes,
        "num_groups" : num_groups,
        "group_mode" : group_mode,
        "ground_truth_w" : ground_truth_w,
        "ground_truth_log_likelihood" : ground_truth_log_likelihood,
        "cost_data" : {}
    }

    w = np.array(json.loads(ground_truth_w), dtype=float)  # Using dtype=float to handle None as NaN

    # ISOLATE TEST FOR ONE SPECIFIC INITIAL COLORING ------------------------

    # initial coloring 2nd eigenvector of A
    # A = nx.adjacency_matrix(graph, nodelist=range(graph.number_of_nodes()))

    # # # Compute the second eigenvector
    # eigvals, eigvecs = eigsh(A, k=2, which='LM')

    # # # Sort eigenvalues in descending order
    # sorted_indices = np.argsort(eigvals)[::-1]  # Indices of eigenvalues sorted from largest to smallest
    # second_largest_index = sorted_indices[1]  # Index of second-largest eigenvalue

    # # Get the corresponding eigenvector
    # second_eigenvector = eigvecs[:, second_largest_index]
    # print(second_eigenvector)

    # # # Assign colors based on the sign of the second-largest eigenvector
    # for i, node in enumerate(range(graph.number_of_nodes())):
    #     graph.nodes[node]['color'] = 0 if second_eigenvector[i] < 0 else 1


    specific_coloring = initial_node_colors[0]
    for node, color in enumerate(specific_coloring):
        graph.nodes[node]['color'] = color

    # **Instantiate GreedyState**
    # greedy_state = SBMState(graph, num_groups, w)
    # graph_g, log_likelihood_data_g, final_w_g, changes_g = greedy_state.optimise(algo_func="reluctant")

    graph_g, log_likelihood_data_g, final_w_g, changes_g = optimise_sbm4(graph, num_groups, group_mode, algo_func=fr)
    # graph_r, log_likelihood_data_r, final_w_r, changes_g = optimise_sbm4(graph, num_groups, group_mode, algo_func=fr)

    # Visualise graph after optimisation
    draw_graph(graph_g, pos=nx.spring_layout(graph, seed=1), graph_name=graph_name, iterations_taken=0, cost_data=log_likelihood_data_g,
            color_set_size=num_groups, 
            degree=None, 
            num_nodes=num_nodes, 
            gaussian_mean=None, 
            gaussian_variance=None,
            ground_truth_log_likelihood = ground_truth_log_likelihood
            )

    # Visualise heatmap of node changes against iterations
    generate_heatmap_of_color_changes(specific_coloring, changes_g, num_nodes, num_groups, color_map)

    # Visualise histogram of number of colorings per node index
    recolor_counts = [0] * num_nodes
    for change in changes_g:
        node_index, _ = change  # We don't need the color, just the node index
        recolor_counts[node_index] += 1
    plt.bar(range(num_nodes), recolor_counts)
    plt.xlabel('Node Index')
    plt.ylabel('Number of Recolors')
    plt.title('Histogram of Node Recolors')
    plt.show()
    # ------------------------------------------------------------------------S

    # # make all colors red which is 0
    # for node in graph_copy.nodes:
    #     graph_copy.nodes[node]['color'] = 0

    # replace all colors with initial colorings

    # start_time = time.time()
    # for i, initial_coloring in enumerate(initial_node_colors):
    #     for node, color in enumerate(initial_coloring):
    #         graph.nodes[node]['color'] = color

    #     graph_copy = graph.copy()
    #     # graph_copy2 = graph.copy()



    #     # **Instantiate class**
    #     greedy_state = SBMState(graph, num_groups, w)
    #     graph_g, log_likelihood_data_g, final_w_g = greedy_state.optimise(algo_func="greedy")
    #     reluctant_state = SBMState(graph_copy, num_groups, w)
    #     graph_r, log_likelihood_data_r, final_w_r = reluctant_state.optimise(algo_func="reluctant")
    #     # reluctant_state = SBMState(graph_copy2, num_groups, w)
    #     # graph_gr, log_likelihood_data_gr, final_w_gr = reluctant_state.optimise(algo_func="greedy_random")

    #     # optimise sbm and get final w and log likelihood
    #     # sbm_graph_g, log_likelihood_data_g, final_w_g = optimise_sbm4(graph, num_groups, group_mode, algo_func=fg)
    #     # sbm_graph_r, log_likelihood_data_r, final_w_r = optimise_sbm4(graph_copy, num_groups, group_mode, algo_func=fr)

    #     results["cost_data"][f"initial_coloring_{i}"] = {
    #         "cost_data_g": log_likelihood_data_g,
    #         "cost_data_r": log_likelihood_data_r
    #         # "cost_data_gr": log_likelihood_data_gr
    #     }

    #     print(f"{i} initial coloring optimisation complete")
    
    # end_time = time.time()
    # print(f"Execution time: {end_time - start_time:.4f} seconds")

    # graphs_path = r"C:\Projects\Heuristics for combinatorial optimisation\Heuristics-for-combinatorial-optimisation\results"

    # with open(os.path.join(graphs_path, f"{graph_name}c_results.json"), 'w') as f:
    #     json.dump(results, f, indent = 2)

    # print(f"Saved results to {graphs_path}/{graph_name}c_results.json")  


    # DONT RECOLOR, INITIAL COLORING IS THE GROUND TRUTH --------------------------------------------------------------------
    # graph_copy = graph.copy()

    # optimise sbm and get final w and log likelihood
    # sbm_graph_g, log_likelihood_data_g, final_w_g = optimise_sbm(graph, num_groups, algo_func="greedy")
    # sbm_graph_r, log_likelihood_data_r, final_w_r = optimise_sbm(graph_copy, num_groups, algo_func="reluctant")

    # print(log_likelihood_data_g)
    # print(log_likelihood_data_g)

    # draw_graph(sbm_graph_g, pos=nx.spring_layout(graph, seed=1), graph_name=graph_name, iterations_taken=0, cost_data=log_likelihood_data_g,
    #         color_set_size=num_groups, 
    #         degree=None, 
    #         num_nodes=num_nodes, 
    #         gaussian_mean=None, 
    #         gaussian_variance=None,
    #         ground_truth_log_likelihood = ground_truth_log_likelihood
    #         )
    # draw_graph(sbm_graph_r, pos=nx.spring_layout(graph, seed=1), graph_name=graph_name, iterations_taken=0, cost_data=log_likelihood_data_r,
    #         color_set_size=num_groups, 
    #         degree=None, 
    #         num_nodes=num_nodes, 
    #         gaussian_mean=None, 
    #         gaussian_variance=None,
    #         ground_truth_log_likelihood = ground_truth_log_likelihood
    #         )
    









    # # draw_graph(sbm_optimised_graph, pos=nx.spring_layout(graph, seed=1), graph_name=graph_name, iterations_taken=0, cost_data=log_likelihood_data,
    # #         color_set_size=num_groups, 
    # #         degree=None, 
    # #         num_nodes=num_nodes, 
    # #         gaussian_mean=None, 
    # #         gaussian_variance=None,
    # #         ground_truth_log_likelihood = ground_truth_log_likelihood
    # #         )
    