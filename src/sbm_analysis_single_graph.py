import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os

def sbm_plot_cost_data(cost_data, graph_name, num_groups, num_nodes, group_mode, ground_truth_w, ground_truth_log_likelihood, specific_coloring):

    plt.figure(figsize=(10, 6))
    # Iterate over each initial coloring in cost_data
    for i, (key, value) in enumerate(cost_data.items()):
        # Skip other colorings if a specific coloring is selected
        if specific_coloring is not None and i != specific_coloring:
            continue

        # Extract data for the greedy and reluctant algorithms
        costs_fg = value["cost_data_g"]
        iterations_fg = list(range(len(costs_fg)))
        costs_fr = value["cost_data_r"]
        iterations_fr = list(range(len(costs_fr)))
        costs_fgr = value["cost_data_gr"]
        iterations_fgr = list(range(len(costs_fgr)))
        costs_frr = value["cost_data_rr"]
        iterations_frr = list(range(len(costs_frr)))

        # Final cost and total iterations for this coloring
        # total_iterations_fg = iterations_fg[-1]
        # final_cost_fg = costs_fg[-1]
        # total_iterations_fr = iterations_fr[-1]
        # final_cost_fr = costs_fr[-1]

        # Plot "Greedy" (fg) with transparency for multiple colorings
        plt.plot(iterations_fg, costs_fg, color="red", alpha=0.6)
        # plt.scatter(total_iterations_fg, final_cost_fg, color="red", s=50, alpha=0.6)

        # Plot "Reluctant" (fr) with transparency for multiple colorings
        plt.plot(iterations_fr, costs_fr, color="green", alpha=0.6)
        # plt.scatter(total_iterations_fr, final_cost_fr, color="green", s=50, alpha=0.6)

        plt.plot(iterations_fgr, costs_fgr, color="orange", alpha=0.6)

        plt.plot(iterations_frr, costs_frr, color="purple", alpha=0.6)

    plt.axhline(y=ground_truth_log_likelihood, color='b', linestyle='--', label='Ground Truth')
    plt.text(0.5, ground_truth_log_likelihood, f'{ground_truth_log_likelihood:.2f}', 
               color='b', ha='center', va='bottom', fontsize=10)

    plt.plot([], [], color="red", label="Greedy")
    plt.plot([], [], color="green", label="Reluctant")
    plt.plot([], [], color="orange", label="Greedy Random")
    plt.plot([], [], color="purple", label="Reluctant Random")
    plt.legend(loc="lower right")
    
    # param_text = (f"Number of Groups: {num_groups}\n"
    #               f"Number of Nodes: {num_nodes}\n"
    #               f"Ground truth w: {ground_truth_w}")
    # plt.gcf().text(0.5, 0.3, param_text, fontsize=8, bbox=dict(facecolor='white', alpha=0.5))

    plt.xlabel("Iterations")
    plt.ylabel("Log Likelihood")
    plt.title(f"LL vs Iterations on {graph_name}")
    plt.grid()

    plt.savefig(f"plots/{graph_name}_cost.png")

    plt.show()

def sbm_plot_final_costs(cost_data, graph_name, num_nodes, num_groups, group_mode, ground_truth_w, ground_truth_log_likelihood):
    '''
    Plot a scatter plot of final log likelihood against initial coloring. 
    Show average final log likelihood.
    Show the ground truth log likelihood
    '''
    greedy_final_costs = []
    reluctant_final_costs = []
    greedy_random_final_costs = []
    reluctant_random_final_costs = []

    # Extract the final cost for each initial coloring for both greedy and reluctant algorithms
    for initial_coloring_key, iteration_data in cost_data.items():
        cost_data_g = iteration_data["cost_data_g"]
        cost_data_r = iteration_data["cost_data_r"]
        cost_data_gr = iteration_data["cost_data_gr"]
        cost_data_rr = iteration_data["cost_data_rr"]

        final_cost_g = cost_data_g[-1]  # Last entry in the greedy cost data
        final_cost_r = cost_data_r[-1]  # Last entry in the reluctant cost data
        final_cost_gr = cost_data_gr[-1]  # Last entry in the greedy random cost data
        final_cost_rr = cost_data_rr[-1]  # Last entry in the greedy random cost data

        greedy_final_costs.append(final_cost_g)
        reluctant_final_costs.append(final_cost_r)
        greedy_random_final_costs.append(final_cost_gr)
        reluctant_random_final_costs.append(final_cost_rr)

    plt.figure(figsize=(10, 6))

    # Scatter plot for greedy and reluctant final costs
    plt.scatter(range(len(greedy_final_costs)), greedy_final_costs, label='Greedy', color='red', alpha=0.6)
    plt.scatter(range(len(reluctant_final_costs)), reluctant_final_costs, label='Reluctant', color='green', alpha=0.6)
    plt.scatter(range(len(greedy_random_final_costs)), greedy_random_final_costs, label='Greedy Random', color='orange', alpha=0.6)
    plt.scatter(range(len(reluctant_random_final_costs)), reluctant_random_final_costs, label='Reluctant Random', color='purple', alpha=0.6)

    # Mean lines for greedy and reluctant
    plt.axhline(np.mean(greedy_final_costs), color='red', linestyle='--', label=f'Mean Greedy: {np.mean(greedy_final_costs)}')
    plt.axhline(np.mean(reluctant_final_costs), color='green', linestyle='--', label=f'Mean Reluctant: {np.mean(reluctant_final_costs)}')
    plt.axhline(np.mean(greedy_random_final_costs), color='orange', linestyle='--', label=f'Mean Greedy Random: {np.mean(greedy_random_final_costs)}')
    plt.axhline(np.mean(reluctant_random_final_costs), color='purple', linestyle='--', label=f'Mean Reluctant Random: {np.mean(reluctant_random_final_costs)}')
    plt.axhline(ground_truth_log_likelihood, color='b', linestyle='--', label=f'Ground Truth: {ground_truth_log_likelihood}')
    plt.text(0.5, ground_truth_log_likelihood, f'{ground_truth_log_likelihood:.2f}', 
               color='b', ha='center', va='bottom', fontsize=10)

    plt.xlabel('Initial Coloring Index')
    plt.ylabel('Final Log Likelihood')
    plt.title(f'Final LL for All Initial Colorings of {graph_name}')

    # experiment_text = (f"Number of Groups: {num_groups}\n"
    #               f"Number of Nodes: {num_nodes}\n"
    #               f"Ground truth w: {ground_truth_w}")
    
    # plt.gca().text(0.05, 0.3, experiment_text, transform=plt.gca().transAxes, fontsize=8, 
    #                verticalalignment='bottom', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.5))

    plt.legend(loc='lower left')

    plt.savefig(f"plots/{graph_name}_scatter.png")

    plt.show()

def sbm_plot_cost_diff_histogram(cost_data, num_nodes, graph_name, num_bins, bin_range):
    '''
    Plot histogram of normalised cost diff for all initial colorings.
    '''
    cost_differences = []
    
    # Loop through all initial colorings and compute the cost difference for each one
    for initial_coloring_key, iteration_data in cost_data.items():
        cost_data_g = iteration_data["cost_data_g"]
        cost_data_r = iteration_data["cost_data_r"]

        # Get the final costs
        final_cost_g = cost_data_g[-1]
        final_cost_r = cost_data_r[-1]

        # Calculate the cost difference between greedy and reluctant
        cost_diff = final_cost_g - final_cost_r

        # Calculate 1/n * cost_diff, where n is the number of nodes
        normalized_cost_diff = (1 / num_nodes) * cost_diff

        # Append to the list
        cost_differences.append(normalized_cost_diff)
    
    # Set bin range if not provided
    if bin_range is None:
        bin_range = (min(cost_differences), max(cost_differences))
    
    plt.figure(figsize=(10, 6))
    plt.hist(cost_differences, bins=np.linspace(bin_range[0], bin_range[1], num_bins), edgecolor='blue', alpha=0.7)
    
    plt.xlabel('Normalized Loglikelihood Difference (1/n * (Greedy - Reluctant))')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Normalized Loglikelihood Differences for {graph_name}')
    
    plt.grid(True)

    plt.savefig(f"plots/{graph_name}_hist.png")

    plt.show()

if __name__ == '__main__':

    # Define variables
    num_nodes = 10000
    num_groups = 2
    group_mode = "t"
    mode_number = 2
    instance_number = 0
    random_prob = 0.15  # Set to None if you don't want to include it

    # file_path
    base_path = r"C:\Projects\Heuristics for combinatorial optimisation\results"
    if group_mode == "t":
        if random_prob is not None:
            file_path = f"SBM({num_nodes}, {num_groups}, {group_mode}{mode_number}{instance_number}, {random_prob})_results.json"
        else:
            file_path = f"SBM({num_nodes}, {num_groups}, {group_mode}{mode_number}{instance_number})_results.json"
    else:
        file_path = f"SBM({num_nodes}, {num_groups}, {group_mode})_results.json"
    file_path = os.path.join(base_path, file_path)

    with open(file_path, 'r') as f:
        data = json.load(f)

    graph_name = data['graph_name']
    if random_prob is not None:
        graph_name = graph_name[:-1] + f", {random_prob})"
    num_nodes = data['num_nodes']
    num_groups = data['num_groups']
    group_mode = data['group_mode']
    ground_truth_w = data['ground_truth_w']
    ground_truth_log_likelihood = data['ground_truth_log_likelihood']
    all_cost_data = data['cost_data']

    # plot log likelihood against iterations for all initial colorings / specific coloring
    sbm_plot_cost_data(all_cost_data, graph_name, num_groups, num_nodes, group_mode, ground_truth_w, ground_truth_log_likelihood, specific_coloring=None)

    # plot scatter of final log likelihood against initial colorings 
    sbm_plot_final_costs(all_cost_data, graph_name, num_nodes, num_groups, group_mode, ground_truth_w, ground_truth_log_likelihood)

    # plot histogram of normalised log likelihood diff for all initial colorings
    sbm_plot_cost_diff_histogram(all_cost_data, num_nodes, graph_name, num_bins=100, bin_range=(-0.2, 0.2))
