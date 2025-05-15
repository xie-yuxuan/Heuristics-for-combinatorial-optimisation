import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os

from sbm_analysis_single_graph import sbm_plot_cost_data, sbm_plot_final_costs

if __name__ == '__main__':

    file_path = r"C:\Projects\Heuristics for combinatorial optimisation\results\wikipedia_map_of_science, 0.05, dynamic_w_results.json"
    
    with open(file_path, 'r') as f:
        data = json.load(f)

    graph_name = data['graph_name']
    # if random_prob is not None:
    #     graph_name = graph_name[:-1] + f", {random_prob})"
    num_nodes = data['num_nodes']
    num_groups = data['num_groups']
    # group_mode = data['group_mode']
    # ground_truth_w = data['ground_truth_w']
    # ground_truth_log_likelihood = data['ground_truth_log_likelihood']
    all_cost_data = data['cost_data']

    # plot log likelihood against iterations for all initial colorings / specific coloring
    sbm_plot_cost_data(all_cost_data, graph_name, num_groups, num_nodes, group_mode=None, ground_truth_w=None, ground_truth_log_likelihood=None, specific_coloring=None)

    # plot scatter of final log likelihood against initial colorings 
    sbm_plot_final_costs(all_cost_data, graph_name, num_nodes, num_groups, group_mode=None, ground_truth_w=None, ground_truth_log_likelihood=None)