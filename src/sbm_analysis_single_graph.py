import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os

from visualisation import plot_cost_data, plot_final_costs, plot_cost_diff_histogram, sbm_plot_cost_data, sbm_plot_final_costs, sbm_plot_cost_diff_histogram

if __name__ == '__main__':

    file_path = r"C:\Projects\Heuristics for combinatorial optimisation\Heuristics-for-combinatorial-optimisation\results\SBM(1000, 4, c)2_results.json"


    with open(file_path, 'r') as f:
        data = json.load(f)


    graph_name = data['graph_name']
    num_nodes = data['num_nodes']
    num_groups = data['num_groups']
    group_mode = data['group_mode']
    ground_truth_log_likelihood = data['ground_truth_log_likelihood']
    all_cost_data = data['cost_data']

    # plot log likelihood against iterations for all initial colorings / specific coloring
    sbm_plot_cost_data(all_cost_data, graph_name, num_groups, num_nodes, group_mode, ground_truth_log_likelihood, specific_coloring=None)

    # plot scatter of final log likelihood against initial colorings 
    sbm_plot_final_costs(all_cost_data, graph_name, num_nodes, num_groups, group_mode, ground_truth_log_likelihood)

    # plot histogram of normalised log likelihood diff for all initial colorings
    sbm_plot_cost_diff_histogram(all_cost_data, num_nodes, graph_name, num_bins=100, bin_range=(-0.2, 0.2))

    # plot_cost_diff_histogram(data["cost_data"], num_nodes, graph_name, num_bins=100, bin_range=(-0.2, 0.2))