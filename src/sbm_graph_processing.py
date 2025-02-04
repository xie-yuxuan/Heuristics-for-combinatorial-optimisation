import json
import os
import numpy as np
import networkx as nx
import time
import copy
from networkx.readwrite import json_graph

from algorithms import optimise_sbm, optimise_sbm2, optimise_sbm3, optimise_sbm4
from visualisation import draw_graph

def load_graph_from_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # load graph_name and attibutes
    graph_name = data["graph_name"]
    num_groups = data["num_groups"]
    num_nodes = data["num_nodes"]
    group_mode = data["group_mode"]
    initial_node_colors = data["initial_node_colors"]
    ground_truth_log_likelihood = data["ground_truth_log_likelihood"]

    graph_data = data["graph_data"]
    graph = json_graph.node_link_graph(graph_data)

    # uncomment to get adj matrix
    # adj_matrix = nx.adjacency_matrix(graph, weight='weight').toarray()
    
    return graph, graph_name, num_nodes, num_groups, group_mode, initial_node_colors, ground_truth_log_likelihood

if __name__ == "__main__":
    file_path = r"C:\Projects\Heuristics for combinatorial optimisation\Heuristics-for-combinatorial-optimisation\data\graphs\SBM(10, 3, a).json"
    graph, graph_name, num_nodes, num_groups, group_mode, initial_node_colors, ground_truth_log_likelihood = load_graph_from_json(file_path)

    results = {
        "graph_name": graph_name,
        "num_nodes" : num_nodes,
        "num_groups" : num_groups,
        "group_mode" : group_mode,
        "ground_truth_log_likelihood" : ground_truth_log_likelihood,
        "cost_data" : {}
    }

    # draw_graph(graph, pos=nx.spring_layout(graph, seed=1), graph_name=graph_name, iterations_taken=0, cost_data=None,
    #         color_set_size=num_groups, 
    #         degree=None, 
    #         num_nodes=num_nodes, 
    #         gaussian_mean=None, 
    #         gaussian_variance=None,
    #         ground_truth_log_likelihood = ground_truth_log_likelihood
    #         ) 

    # ISOLATE TEST FOR ONE SPECIFIC INITIAL COLORING ------------------------

    for node, color in enumerate(initial_node_colors[0]):
        graph.nodes[node]['color'] = color

    graph_g, log_likelihood_data_g, final_w_g = optimise_sbm2(graph, num_groups, group_mode, algo_func="greedy")
    # graph_r, log_likelihood_data_r, final_w_r = optimise_sbm4(graph, num_groups, group_mode, algo_func="reluctant")

    draw_graph(graph_g, pos=nx.spring_layout(graph, seed=1), graph_name=graph_name, iterations_taken=0, cost_data=log_likelihood_data_g,
            color_set_size=num_groups, 
            degree=None, 
            num_nodes=num_nodes, 
            gaussian_mean=None, 
            gaussian_variance=None,
            ground_truth_log_likelihood = ground_truth_log_likelihood
            )

    # print(log_likelihood_data_g)
    # ------------------------------------------------------------------------S

    # # make all colors red which is 0
    # for node in graph_copy.nodes:
    #     graph_copy.nodes[node]['color'] = 0

    # replace all colors with initial colorings
    # start_time = time.time()
    # for i, initial_coloring in enumerate(initial_node_colors):
    #     for node, color in enumerate(initial_coloring):
    #         graph.nodes[node]['color'] = color

    # # # draw_graph(graph, pos=nx.spring_layout(graph, seed=1), graph_name=graph_name, iterations_taken=0, cost_data=None,
    # # #         color_set_size=num_groups, 
    # # #         degree=None, 
    # # #         num_nodes=num_nodes, 
    # # #         gaussian_mean=None, 
    # # #         gaussian_variance=None,
    # # #         ground_truth_log_likelihood = ground_truth_log_likelihood
    # # #         )

    #     graph_copy = graph.copy()

    #     # optimise sbm and get final w and log likelihood
    #     sbm_graph_g, log_likelihood_data_g, final_w_g = optimise_sbm3(graph, num_groups, group_mode, algo_func="greedy")
    #     sbm_graph_r, log_likelihood_data_r, final_w_r = optimise_sbm3(graph_copy, num_groups, group_mode, algo_func="reluctant")

    #     results["cost_data"][f"initial_coloring_{i}"] = {
    #         "cost_data_g": log_likelihood_data_g,
    #         "cost_data_r": log_likelihood_data_r
    #     }

    #     print(f"{i} initial coloring optimisation complete")
    
    # end_time = time.time()
    # print(f"Execution time: {end_time - start_time:.4f} seconds")

    # graphs_path = r"C:\Projects\Heuristics for combinatorial optimisation\Heuristics-for-combinatorial-optimisation\results"

    # with open(os.path.join(graphs_path, f"{graph_name}3_results.json"), 'w') as f:
    #     json.dump(results, f, indent = 2)

    # print(f"Saved results to {graphs_path}/{graph_name}3_results.json")  


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
    