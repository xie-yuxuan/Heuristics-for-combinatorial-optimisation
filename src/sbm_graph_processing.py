import json
import os
import numpy as np
import networkx as nx
import time
import copy
from networkx.readwrite import json_graph

from algorithms import optimise_sbm2
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
    file_path = r"C:\Projects\Heuristics for combinatorial optimisation\Heuristics-for-combinatorial-optimisation\data\graphs\SBM(40, 4, b).json"
    graph, graph_name, num_nodes, num_groups, group_mode, initial_node_colors, ground_truth_log_likelihood = load_graph_from_json(file_path)

    # replace all colors with initial colorings
    graph_copy = graph.copy()

    # # make all colors red which is 0
    # for node in graph_copy.nodes:
    #     graph_copy.nodes[node]['color'] = 0

    # draw_graph(graph_copy, pos=nx.spring_layout(graph, seed=1), graph_name=graph_name, iterations_taken=0, cost_data=None,
    #         color_set_size=num_groups, 
    #         degree=None, 
    #         num_nodes=num_nodes, 
    #         gaussian_mean=None, 
    #         gaussian_variance=None,
    #         ground_truth_log_likelihood = ground_truth_log_likelihood
    #         )

    for node, color in enumerate(initial_node_colors[0]):
        graph_copy.nodes[node]['color'] = color

    # optimise sbm and get final w and log likelihood
    sbm_optimised_graph, log_likelihood_data, final_w = optimise_sbm2(graph_copy, num_groups, algo_func=None)
    print(final_w, log_likelihood_data)

    draw_graph(sbm_optimised_graph, pos=nx.spring_layout(graph, seed=1), graph_name=graph_name, iterations_taken=0, cost_data=log_likelihood_data,
            color_set_size=num_groups, 
            degree=None, 
            num_nodes=num_nodes, 
            gaussian_mean=None, 
            gaussian_variance=None,
            ground_truth_log_likelihood = ground_truth_log_likelihood
            )
    