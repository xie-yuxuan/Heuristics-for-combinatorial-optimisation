import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
from scipy import stats

def sbm_plot_cost_data(cost_data, graph_name, num_groups, num_nodes, group_mode, ground_truth_w, ground_truth_log_likelihood, specific_coloring):
    plt.rcParams.update({
        "font.size": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "figure.titlesize": 16
    })

    plt.figure(figsize=(10, 6))

    for i, (key, value) in enumerate(cost_data.items()):
        if specific_coloring is not None and i != specific_coloring:
            continue
        costs_fg = value["cost_data_g"]
        costs_fr = value["cost_data_r"]
        # costs_fgr = value["cost_data_gr"]
        # costs_frr = value["cost_data_rr"]
        # costs_fgsa = value["cost_data_gsa"]
        # costs_frsa = value["cost_data_rsa"]

        iterations_fg = list(range(len(costs_fg)))
        iterations_fr = list(range(len(costs_fr)))
        # iterations_fgr = list(range(len(costs_fgr)))
        # iterations_frr = list(range(len(costs_frr)))
        # iterations_fgsa = list(range(len(costs_fgsa)))
        # iterations_frsa = list(range(len(costs_frsa)))

        plt.plot(iterations_fg, costs_fg, color="red", alpha=0.6)
        plt.plot(iterations_fr, costs_fr, color="green", alpha=0.6)
        # plt.plot(iterations_fgr, costs_fgr, color="orange", alpha=0.6)
        # plt.plot(iterations_frr, costs_frr, color="purple", alpha=0.6)
        # plt.plot(iterations_fgsa, costs_fgsa, color="brown", alpha=0.6)
        # plt.plot(iterations_frsa, costs_frsa, color="blue", alpha=0.6)

    if ground_truth_log_likelihood is not None:
        plt.axhline(y=ground_truth_log_likelihood, color='b', linestyle='--', label='Ground Truth')
        plt.text(0.5, ground_truth_log_likelihood, f'{ground_truth_log_likelihood:.2f}',
                 color='b', ha='center', va='bottom', fontsize=10)

    # Dummy handles for legend
    plt.plot([], [], color="red", label="Greedy")
    plt.plot([], [], color="green", label="Reluctant")
    # plt.plot([], [], color="orange", label="Greedy Random")
    # plt.plot([], [], color="purple", label="Reluctant Random")
    # plt.plot([], [], color="brown", label="Greedy SA")
    # plt.plot([], [], color="blue", label="Reluctant SA")
    plt.legend(loc="lower right")

    print("ground truth w", ground_truth_w)
    print("ground truth log likelihood", ground_truth_log_likelihood)

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
    Show the ground truth log likelihood.
    '''
    plt.rcParams.update({
        "font.size": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "figure.titlesize": 16
    })
    greedy_final_costs = []
    reluctant_final_costs = []
    greedy_random_final_costs = []
    reluctant_random_final_costs = []
    greedy_sa_final_costs = []
    reluctant_sa_final_costs = []

    for _, iteration_data in cost_data.items():
        greedy_final_costs.append(iteration_data["cost_data_g"][-1])
        reluctant_final_costs.append(iteration_data["cost_data_r"][-1])
        greedy_random_final_costs.append(iteration_data["cost_data_gr"][-1])
        # reluctant_random_final_costs.append(iteration_data["cost_data_rr"][-1])
        # greedy_sa_final_costs.append(iteration_data["cost_data_gsa"][-1])
        # reluctant_sa_final_costs.append(iteration_data["cost_data_rsa"][-1])

    plt.figure(figsize=(10, 6))

    indices = range(len(greedy_final_costs))

    # Scatter plot
    plt.scatter(indices, greedy_final_costs, label='Greedy', color='red', alpha=0.6)
    plt.scatter(indices, reluctant_final_costs, label='Reluctant', color='green', alpha=0.6)
    plt.scatter(indices, greedy_random_final_costs, label='Greedy Random', color='orange', alpha=0.6)
    # plt.scatter(indices, reluctant_random_final_costs, label='Reluctant Random', color='purple', alpha=0.6)
    # plt.scatter(indices, greedy_sa_final_costs, label='Greedy SA', color='brown', alpha=0.6)
    # plt.scatter(indices, reluctant_sa_final_costs, label='Reluctant SA', color='blue', alpha=0.6)

    # Mean lines
    plt.axhline(np.mean(greedy_final_costs), color='red', linestyle='--', label=f'Mean Greedy: {np.mean(greedy_final_costs):.2f}')
    plt.axhline(np.mean(reluctant_final_costs), color='green', linestyle='--', label=f'Mean Reluctant: {np.mean(reluctant_final_costs):.2f}')
    plt.axhline(np.mean(greedy_random_final_costs), color='orange', linestyle='--', label=f'Mean Greedy Random: {np.mean(greedy_random_final_costs):.2f}')
    # plt.axhline(np.mean(reluctant_random_final_costs), color='purple', linestyle='--', label=f'Mean Reluctant Random: {np.mean(reluctant_random_final_costs):.2f}')
    # plt.axhline(np.mean(greedy_sa_final_costs), color='brown', linestyle='--', label=f'Mean Greedy SA: {np.mean(greedy_sa_final_costs):.2f}')
    # plt.axhline(np.mean(reluctant_sa_final_costs), color='blue', linestyle='--', label=f'Mean Reluctant SA: {np.mean(reluctant_sa_final_costs):.2f}')

    # Ground truth line
    if ground_truth_log_likelihood is not None:
        plt.axhline(ground_truth_log_likelihood, color='black', linestyle='-.', label=f'Ground Truth: {ground_truth_log_likelihood:.2f}')
        plt.text(0.5, ground_truth_log_likelihood, f'{ground_truth_log_likelihood:.2f}', 
                 color='black', ha='center', va='bottom', fontsize=14)

    plt.xlabel('Initial Coloring Index')
    plt.ylabel('Final Log Likelihood')
    plt.title(f'Final LL for All Initial Colorings of {graph_name}')
    plt.legend(loc='lower left')
    plt.grid()

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

def compute_sbm_summary_table(cost_data, num_nodes, graph_name):
    greedy_random_costs = []
    reluctant_random_costs = []
    greedy_random_iters = []
    reluctant_random_iters = []
    greedy_sa_iters = []
    reluctant_sa_iters = []

    for run in cost_data.values():
        try:
            if "cost_data_gr" in run:
                gr_final = run["cost_data_gr"][-1]
                if isinstance(gr_final, (list, np.ndarray)):
                    gr_final = gr_final[-1]
                greedy_random_costs.append(float(gr_final))
                greedy_random_iters.append(len(run["cost_data_gr"]))

            if "cost_data_rr" in run:
                rr_final = run["cost_data_rr"][-1]
                if isinstance(rr_final, (list, np.ndarray)):
                    rr_final = rr_final[-1]
                reluctant_random_costs.append(float(rr_final))
                reluctant_random_iters.append(len(run["cost_data_rr"]))

            if "cost_data_gsa" in run:
                greedy_sa_iters.append(len(run["cost_data_gsa"]))
            if "cost_data_rsa" in run:
                reluctant_sa_iters.append(len(run["cost_data_rsa"]))

        except Exception as e:
            print(f"Skipping malformed entry due to error: {e}")

    greedy_random_costs = np.array(greedy_random_costs)
    reluctant_random_costs = np.array(reluctant_random_costs)
    greedy_random_iters = np.array(greedy_random_iters)
    reluctant_random_iters = np.array(reluctant_random_iters)
    greedy_sa_iters = np.array(greedy_sa_iters)
    reluctant_sa_iters = np.array(reluctant_sa_iters)

    def mean_ci(arr):
        mean = np.mean(arr)
        ci = 1.96 * np.std(arr, ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0
        return mean, ci

    print(f"--- SBM Summary for {graph_name} (n={num_nodes}) ---")

    if len(greedy_random_costs):
        gr_mean, gr_ci = mean_ci(greedy_random_costs)
        print(f"Mean Final Cost (Greedy Random): {gr_mean:.2f} ± {gr_ci:.2f}")
    if len(reluctant_random_costs):
        rr_mean, rr_ci = mean_ci(reluctant_random_costs)
        print(f"Mean Final Cost (Reluctant Random): {rr_mean:.2f} ± {rr_ci:.2f}")
    if len(greedy_random_costs) and len(reluctant_random_costs):
        cost_diff = (greedy_random_costs - reluctant_random_costs) / num_nodes
        diff_mean, diff_ci = mean_ci(cost_diff)
        print(f"Normalised Cost Diff (GR - RR) / n: {diff_mean:.6f} ± {diff_ci:.6f}")

    if len(greedy_random_iters):
        gr_iter_mean, gr_iter_ci = mean_ci(greedy_random_iters)
        print(f"Mean Iterations (Greedy Random): {gr_iter_mean:.2f} ± {gr_iter_ci:.2f}")
    if len(reluctant_random_iters):
        rr_iter_mean, rr_iter_ci = mean_ci(reluctant_random_iters)
        print(f"Mean Iterations (Reluctant Random): {rr_iter_mean:.2f} ± {rr_iter_ci:.2f}")

    if len(greedy_sa_iters):
        gsa_mean, gsa_ci = mean_ci(greedy_sa_iters)
        print(f"Mean Iterations (Greedy SA): {gsa_mean:.2f} ± {gsa_ci:.2f}")
    if len(reluctant_sa_iters):
        rsa_mean, rsa_ci = mean_ci(reluctant_sa_iters)
        print(f"Mean Iterations (Reluctant SA): {rsa_mean:.2f} ± {rsa_ci:.2f}")

if __name__ == '__main__':

    # Define variables
    num_nodes = 10000
    num_groups = 2
    group_mode = "t"
    mode_number = 7
    instance_number = 0
    random_prob = 0.51 # Set to None if you don't want to include it

    # file_path
    base_path = r"C:\Projects\Heuristics for combinatorial optimisation\results"
    if group_mode == "t":
        if random_prob is not None:
            file_path = f"SBM({num_nodes}, {num_groups}, {group_mode}{mode_number}{instance_number}, {random_prob})_results.json"
        else:
            file_path = f"SBM({num_nodes}, {num_groups}, {group_mode}{mode_number}{instance_number})_results.json"
    else:
        file_path = f"SBM({num_nodes}, {num_groups}, {group_mode}, {random_prob})_results.json"
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
    # sbm_plot_cost_diff_histogram(all_cost_data, num_nodes, graph_name, num_bins=100, bin_range=(-0.2, 0.2))

    # compute_sbm_summary_table(all_cost_data, num_nodes, graph_name)
