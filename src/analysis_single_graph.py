import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
from scipy import stats


'''
Run this program to run an analysis for a single graph, returning results and showing plots.
'''




def get_best_final_cost(cost_data):
    '''
    Return best final cost and corresponding number of iterations.
    Purpose is to find global optima.
    '''
    min_final_cost_g = float('inf')
    min_final_cost_r = float('inf')
    min_iterations_fg = None
    min_iterations_fr = None

    # Iterate through all initial colorings in the cost data
    for value in cost_data.values():
        # Extract cost data for greedy and reluctant approaches
        costs_fg = value["cost_data_g"]
        iterations_fg = list(range(len(costs_fg)))
        costs_fr = value["cost_data_r"]
        iterations_fr = list(range(len(costs_fr)))

        # Get the final cost and total iterations for each approach
        final_cost_fg = costs_fg[-1]
        total_iterations_fg = iterations_fg[-1]
        final_cost_fr = costs_fr[-1]
        total_iterations_fr = iterations_fr[-1]

        # Update minimum final cost and iterations for Greedy
        if final_cost_fg < min_final_cost_g:
            min_final_cost_g = final_cost_fg
            min_iterations_fg = total_iterations_fg

        # Update minimum final cost and iterations for Reluctant
        if final_cost_fr < min_final_cost_r:
            min_final_cost_r = final_cost_fr
            min_iterations_fr = total_iterations_fr

    return {
        "Greedy": {"Best Final Cost": min_final_cost_g, "Iterations": min_iterations_fg},
        "Reluctant": {"Best Final Cost": min_final_cost_r, "Iterations": min_iterations_fr}
    }


def calculate_greedy_vs_reluctant_stats(cost_data):
    '''
    Return avg final cost (and avg cost diff), std dev, and probability that greedy is better.
    '''
    greedy_better_count = 0
    cost_differences = []
    greedy_final_costs = []
    reluctant_final_costs = []

    # Loop through all initial colorings
    for initial_coloring_key, iteration_data in cost_data.items():
        cost_data_g = iteration_data["cost_data_g"]
        cost_data_r = iteration_data["cost_data_r"]

        final_cost_g = cost_data_g[-1]  # Last entry in the greedy cost data
        final_cost_r = cost_data_r[-1]  # Last entry in the reluctant cost data

        greedy_final_costs.append(final_cost_g)
        reluctant_final_costs.append(final_cost_r)

        # Track if greedy is better
        if final_cost_g < final_cost_r:
            greedy_better_count += 1

        # Calculate the cost difference
        cost_difference = final_cost_g - final_cost_r
        cost_differences.append(cost_difference)

    # Calculate probability that greedy is better
    probability_greedy_better = greedy_better_count / len(cost_differences)

    # Calculate the average cost difference between greedy and reluctant
    avg_cost_difference = np.mean(cost_differences)

    # Calculate the average final costs for both greedy and reluctant
    avg_greedy_final_cost = np.mean(greedy_final_costs)
    avg_reluctant_final_cost = np.mean(reluctant_final_costs)

    # Calculate the standard deviation for both greedy and reluctant final costs
    std_greedy_final_cost = np.std(greedy_final_costs)
    std_reluctant_final_cost = np.std(reluctant_final_costs)

    # Calculate the standard deviation for the cost differences
    std_cost_difference = np.std(cost_differences)

    # Calculate ±2σ for both greedy and reluctant final costs, and cost differences
    greedy_2sigma_range = (avg_greedy_final_cost - 2 * std_greedy_final_cost, avg_greedy_final_cost + 2 * std_greedy_final_cost)
    reluctant_2sigma_range = (avg_reluctant_final_cost - 2 * std_reluctant_final_cost, avg_reluctant_final_cost + 2 * std_reluctant_final_cost)
    cost_diff_2sigma_range = (avg_cost_difference - 2 * std_cost_difference, avg_cost_difference + 2 * std_cost_difference)

    return {
        "probability_greedy_better": probability_greedy_better,
        "avg_cost_difference": avg_cost_difference,
        "avg_greedy_final_cost": avg_greedy_final_cost,
        "avg_reluctant_final_cost": avg_reluctant_final_cost,
        "greedy_2sigma_range": greedy_2sigma_range,
        "reluctant_2sigma_range": reluctant_2sigma_range,
        "cost_diff_2sigma_range": cost_diff_2sigma_range
    }

def avg_norm_cost_diff(cost_data, num_nodes):
    cost_differences = []

    # Loop through all initial colorings
    for initial_coloring_key, iteration_data in cost_data.items():
        cost_data_g = iteration_data["cost_data_g"]
        cost_data_r = iteration_data["cost_data_r"]

        final_cost_g = cost_data_g[-1]  # Last entry in the greedy cost data
        final_cost_r = cost_data_r[-1]  # Last entry in the reluctant cost data

        # Calculate the cost difference
        cost_difference = final_cost_g - final_cost_r
        cost_differences.append(cost_difference)

    # Calculate the average and standard deviation of the cost differences
    avg_cost_difference = np.mean(cost_differences)
    std_cost_difference = np.std(cost_differences)

    return avg_cost_difference / num_nodes, std_cost_difference / num_nodes

def plot_cost_data(cost_data, graph_name, color_set_size, degree, num_nodes, gaussian_mean, gaussian_variance, specific_coloring=None):
    '''
    Plot cost against iterations for one graph, for Greedy, Reluctant, and their random variants.
    '''

    plt.rcParams.update({
        "font.size": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 12,
        "figure.titlesize": 16
    })

    plt.figure(figsize=(10, 6))

    # Track min costs and iterations for annotations
    min_data = {
        "g": {"cost": float('inf'), "iter": None, "color": "red"},
        "r": {"cost": float('inf'), "iter": None, "color": "green"},
        "gr": {"cost": float('inf'), "iter": None, "color": "orange"},
        "rr": {"cost": float('inf'), "iter": None, "color": "purple"},
        "gsa": {"cost": float('inf'), "iter": None, "color": "brown"},
        "rsa": {"cost": float('inf'), "iter": None, "color": "blue"}
    }

    for i, (key, value) in enumerate(cost_data.items()):
        if specific_coloring is not None and i != specific_coloring:
            continue

        for label, color, field in [
            ("g", "red", "cost_data_g"),
                                    ("r", "green", "cost_data_r"),
                                    ("gr", "orange", "cost_data_gr"),
                                    ("rr", "purple", "cost_data_rr"),
                                    ("gsa", "brown", "cost_data_gsa"),
                                    ("rsa", "blue", "cost_data_rsa")
                                    ]:
            
            if field not in value:
                continue
            costs = value[field]
            iterations = list(range(len(costs)))

            plt.plot(iterations, costs, color=color, alpha=0.6)

            final_cost = costs[-1]
            final_iter = iterations[-1]
            if final_cost < min_data[label]["cost"]:
                min_data[label]["cost"] = final_cost
                min_data[label]["iter"] = final_iter

    # Add annotations
    for label, meta in min_data.items():
        if meta["iter"] is not None:
            plt.annotate(f"Iter: {meta['iter']}\nCost: {meta['cost']:.2f}",
                         (meta["iter"], meta["cost"]),
                         textcoords="offset points",
                         xytext=(10, -15),
                         ha='center',
                         color=meta["color"],
                         fontsize=12)

    # Dummy legend handles
    plt.plot([], [], color="red", label="Greedy")
    plt.plot([], [], color="green", label="Reluctant")
    plt.plot([], [], color="orange", label="Greedy Random")
    plt.plot([], [], color="purple", label="Reluctant Random")
    plt.plot([], [], color="brown", label="Greedy SA")
    plt.plot([], [], color="blue", label="Reluctant SA")
    plt.legend(loc="upper right")

    param_text = (f"Color Set Size: {color_set_size}\n"
                  f"Degree: {degree}\n"
                  f"Number of Nodes: {num_nodes}\n"
                  f"Gaussian Mean: {gaussian_mean}\n"
                  f"Gaussian Variance: {gaussian_variance}")
    plt.gcf().text(0.5, 0.85, param_text, fontsize=12, ha='center', va='top', bbox=dict(facecolor='white', alpha=0.5))

    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title(f"Cost vs Iterations for All Algorithms on {graph_name}")
    plt.grid()

    plt.savefig(f"plots/{graph_name}_cost.png")
    plt.show()

# def plot_cost_data(cost_data, graph_name, color_set_size, degree, num_nodes, gaussian_mean, gaussian_variance, specific_coloring=None):
#     '''
#     Plot cost against iterations for one graph, for Greedy, Reluctant, and their random variants.
#     Compatible with both old (2D list: [iterations, costs]) and new (1D list: costs) formats.
#     '''
#     plt.rcParams.update({
#         "font.size": 16,
#         "axes.labelsize": 16,
#         "xtick.labelsize": 16,
#         "ytick.labelsize": 16,
#         "legend.fontsize": 14,
#         "figure.titlesize": 16
#     })

#     plt.figure(figsize=(10, 6))

#     min_data = {
#         "g": {"cost": float('inf'), "iter": None, "color": "red"},
#         "r": {"cost": float('inf'), "iter": None, "color": "green"},
#     }

#     for i, (key, value) in enumerate(cost_data.items()):
#         if specific_coloring is not None and i != specific_coloring:
#             continue

#         for label, color, field in [
#             ("g", "red", "cost_data_g"),
#             ("r", "green", "cost_data_r"),
#         ]:
#             if field not in value:
#                 continue

#             raw = value[field]

#             # Handle old format [iterations, costs]
#             if isinstance(raw, list) and len(raw) == 2 and isinstance(raw[0], list):
#                 iterations, costs = raw
#             else:
#                 costs = raw
#                 iterations = list(range(len(costs)))

#             plt.plot(iterations, costs, color=color, alpha=0.6)

#             final_cost = costs[-1]
#             final_iter = iterations[-1]
#             if final_cost < min_data[label]["cost"]:
#                 min_data[label]["cost"] = final_cost
#                 min_data[label]["iter"] = final_iter

#     # Annotations
#     for label, meta in min_data.items():
#         if meta["iter"] is not None:
#             plt.annotate(f"Iter: {meta['iter']}\nCost: {meta['cost']:.2f}",
#                          (meta["iter"], meta["cost"]),
#                          textcoords="offset points",
#                          xytext=(10, -15),
#                          ha='center',
#                          color=meta["color"],
#                          fontsize=14)

#     # Legend
#     plt.plot([], [], color="red", label="Greedy")
#     plt.plot([], [], color="green", label="Reluctant")
#     plt.legend(loc="upper right")

#     # Meta-text
#     param_text = (f"Color Set Size: {color_set_size}\n"
#                   f"Degree: {degree}\n"
#                   f"Number of Nodes: {num_nodes}\n"
#                   f"Gaussian Mean: {gaussian_mean}\n"
#                   f"Gaussian Variance: {gaussian_variance}")
#     plt.gcf().text(0.5, 0.85, param_text, fontsize=14, ha='center', va='top', bbox=dict(facecolor='white', alpha=0.5))

#     plt.xlabel("Iterations")
#     plt.ylabel("Cost")
#     plt.title(f"Cost vs Iterations for All Algorithms on {graph_name}")
#     plt.grid()

#     plt.savefig(f"plots/{graph_name}_cost.png")
#     plt.show()



def plot_final_costs(cost_data, graph_name, degree, num_nodes, color_set_size, gaussian_mean, gaussian_variance):
    '''
    Plot a scatter plot of final costs against initial coloring.
    Show average final cost for Greedy, Reluctant, and their random variants.
    '''
    plt.rcParams.update({
        "font.size": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 12,
        "figure.titlesize": 16
    })
    greedy_final_costs = []
    reluctant_final_costs = []
    greedy_random_final_costs = []
    reluctant_random_final_costs = []
    greedy_sa_final_costs = []
    reluctant_sa_final_costs = []

    for initial_coloring_key, iteration_data in cost_data.items():
        if "cost_data_g" in iteration_data:
            greedy_final_costs.append(iteration_data["cost_data_g"][-1])
        if "cost_data_r" in iteration_data:
            reluctant_final_costs.append(iteration_data["cost_data_r"][-1])
        if "cost_data_gr" in iteration_data:
            greedy_random_final_costs.append(iteration_data["cost_data_gr"][-1])
        if "cost_data_rr" in iteration_data:
            reluctant_random_final_costs.append(iteration_data["cost_data_rr"][-1])
        if "cost_data_gsa" in iteration_data:
            greedy_sa_final_costs.append(iteration_data["cost_data_gsa"][-1])
        if "cost_data_rsa" in iteration_data:
            reluctant_sa_final_costs.append(iteration_data["cost_data_rsa"][-1])


    plt.figure(figsize=(10, 6))

    # Plot final costs
    if greedy_final_costs:
        plt.scatter(range(len(greedy_final_costs)), greedy_final_costs, label='Greedy', color='red', alpha=0.6)
        plt.axhline(np.mean(greedy_final_costs), color='red', linestyle='--', label=f'Mean Greedy: {np.mean(greedy_final_costs):.2f}')
    if reluctant_final_costs:
        plt.scatter(range(len(reluctant_final_costs)), reluctant_final_costs, label='Reluctant', color='green', alpha=0.6)
        plt.axhline(np.mean(reluctant_final_costs), color='green', linestyle='--', label=f'Mean Reluctant: {np.mean(reluctant_final_costs):.2f}')
    if greedy_random_final_costs:
        plt.scatter(range(len(greedy_random_final_costs)), greedy_random_final_costs, label='Greedy Random', color='orange', alpha=0.6)
        plt.axhline(np.mean(greedy_random_final_costs), color='orange', linestyle='--', label=f'Mean Greedy Random: {np.mean(greedy_random_final_costs):.2f}')
    if reluctant_random_final_costs:
        plt.scatter(range(len(reluctant_random_final_costs)), reluctant_random_final_costs, label='Reluctant Random', color='purple', alpha=0.6)
        plt.axhline(np.mean(reluctant_random_final_costs), color='purple', linestyle='--', label=f'Mean Reluctant Random: {np.mean(reluctant_random_final_costs):.2f}')
    if greedy_sa_final_costs:
        plt.scatter(range(len(greedy_sa_final_costs)), greedy_sa_final_costs, label='Greedy SA', color='brown', alpha=0.6)
        plt.axhline(np.mean(greedy_sa_final_costs), color='brown', linestyle='--', label=f'Mean Greedy SA: {np.mean(greedy_sa_final_costs):.2f}')
    if reluctant_sa_final_costs:
        plt.scatter(range(len(reluctant_sa_final_costs)), reluctant_sa_final_costs, label='Reluctant SA', color='blue', alpha=0.6)
        plt.axhline(np.mean(reluctant_sa_final_costs), color='blue', linestyle='--', label=f'Mean Reluctant SA: {np.mean(reluctant_sa_final_costs):.2f}')

    # Labels and styling
    plt.xlabel('Initial Coloring Index')
    plt.ylabel('Final Cost')
    plt.title(f'Final Costs for All Initial Colorings of {graph_name}')

    # experiment_text = (f"Degree: {degree}\nNum Nodes: {num_nodes}\nColor Set Size: {color_set_size}\n"
    #                    f"Gaussian Mean: {gaussian_mean}\nGaussian Variance: {gaussian_variance}")
    # plt.gca().text(0.95, 0.8, experiment_text, transform=plt.gca().transAxes, fontsize=14, 
    #                verticalalignment='bottom', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5))

    plt.legend(loc='upper left')
    plt.savefig(f"plots/{graph_name}_scatter.png")
    plt.grid()
    plt.show()

def plot_cost_diff_histogram(cost_data, num_nodes, graph_name, num_bins, bin_range):
    '''
    Plot histogram of normalised cost diff for all initial colorings.
    '''
    cost_differences = []
    
    # Loop through all initial colorings and compute the cost difference for each one
    for initial_coloring_key, iteration_data in cost_data.items():
        cost_data_g = iteration_data["cost_data_g"]
        cost_data_r = iteration_data["cost_data_r"]

        # Get the final costs
        final_cost_g = cost_data_g[-1][-1]
        final_cost_r = cost_data_r[-1][-1]

        # Calculate the cost difference between greedy and reluctant
        cost_diff = final_cost_g - final_cost_r

        # Calculate 1/n * cost_diff, where n is the number of nodes
        normalized_cost_diff = (1 / num_nodes) * cost_diff

        # Append to the list
        cost_differences.append(normalized_cost_diff)

    # Compute statistics
    mean_diff = np.mean(cost_differences)
    var_diff = np.var(cost_differences)

    print(f"Graph: {graph_name}")
    print(f"Mean of normalised cost differences: {mean_diff:.6f}")
    print(f"Variance of normalised cost differences: {var_diff:.6f}")

    
    # Set bin range if not provided
    if bin_range is None:
        bin_range = (min(cost_differences), max(cost_differences))
    
    plt.figure(figsize=(10, 6))
    plt.hist(cost_differences, bins=np.linspace(bin_range[0], bin_range[1], num_bins), edgecolor='blue', alpha=0.7)
    
    plt.xlabel('Normalized Cost Difference (1/n * (Greedy - Reluctant))')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Normalized Cost Differences for {graph_name}')
    
    plt.grid(True)

    # plt.savefig(f"plots/{graph_name}_hist.png")

    plt.show()

# def compute_summary_table(cost_data, num_nodes, graph_name):
#     greedy_random_costs = []
#     reluctant_random_costs = []
#     greedy_random_iters = []
#     reluctant_random_iters = []
#     greedy_sa_costs = []
#     reluctant_sa_costs = []
#     greedy_sa_iters = []
#     reluctant_sa_iters = []

#     for run in cost_data.values():
#         try:
#             if "cost_data_gr" in run and "cost_data_rr" in run:
#                 gr_final = run["cost_data_gr"][-1]
#                 rr_final = run["cost_data_rr"][-1]

#                 if isinstance(gr_final, (list, np.ndarray)):
#                     gr_final = gr_final[-1]
#                 if isinstance(rr_final, (list, np.ndarray)):
#                     rr_final = rr_final[-1]

#                 greedy_random_costs.append(float(gr_final))
#                 reluctant_random_costs.append(float(rr_final))
#                 greedy_random_iters.append(len(run["cost_data_gr"]))
#                 reluctant_random_iters.append(len(run["cost_data_rr"]))

#             if "cost_data_gsa" in run:
#                 gsa_final = run["cost_data_gsa"][-1]
#                 if isinstance(gsa_final, (list, np.ndarray)):
#                     gsa_final = gsa_final[-1]
#                 greedy_sa_costs.append(float(gsa_final))
#                 greedy_sa_iters.append(len(run["cost_data_gsa"]))

#             if "cost_data_rsa" in run:
#                 rsa_final = run["cost_data_rsa"][-1]
#                 if isinstance(rsa_final, (list, np.ndarray)):
#                     rsa_final = rsa_final[-1]
#                 reluctant_sa_costs.append(float(rsa_final))
#                 reluctant_sa_iters.append(len(run["cost_data_rsa"]))

#         except Exception as e:
#             print(f"Skipping malformed entry due to error: {e}")

#     greedy_random_costs = np.array(greedy_random_costs)
#     reluctant_random_costs = np.array(reluctant_random_costs)
#     greedy_random_iters = np.array(greedy_random_iters)
#     reluctant_random_iters = np.array(reluctant_random_iters)
#     greedy_sa_costs = np.array(greedy_sa_costs)
#     reluctant_sa_costs = np.array(reluctant_sa_costs)
#     greedy_sa_iters = np.array(greedy_sa_iters)
#     reluctant_sa_iters = np.array(reluctant_sa_iters)

#     def mean_ci(arr):
#         mean = np.mean(arr)
#         ci = 1.96 * np.std(arr, ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0
#         return mean, ci

#     gr_mean, gr_ci = mean_ci(greedy_random_costs)
#     rr_mean, rr_ci = mean_ci(reluctant_random_costs)
#     cost_diff = (greedy_random_costs - reluctant_random_costs) / num_nodes
#     diff_mean, diff_ci = mean_ci(cost_diff)
#     gr_iter_mean, gr_iter_ci = mean_ci(greedy_random_iters)
#     rr_iter_mean, rr_iter_ci = mean_ci(reluctant_random_iters)

#     print(f"--- Summary for {graph_name} (n={num_nodes}) ---")
#     print(f"Mean Final Cost (Greedy Random): {gr_mean:.2f} ± {gr_ci:.2f}")
#     print(f"Mean Final Cost (Reluctant Random): {rr_mean:.2f} ± {rr_ci:.2f}")
#     print(f"Normalised Cost Difference (GR - RR): {diff_mean:.4f} ± {diff_ci:.4f}")
#     print(f"Mean Iterations (Greedy Random): {gr_iter_mean:.2f} ± {gr_iter_ci:.2f}")
#     print(f"Mean Iterations (Reluctant Random): {rr_iter_mean:.2f} ± {rr_iter_ci:.2f}")

#     if len(greedy_sa_costs):
#         gsa_cost_mean, gsa_cost_ci = mean_ci(greedy_sa_costs)
#         gsa_iter_mean, gsa_iter_ci = mean_ci(greedy_sa_iters)
#         print(f"Mean Final Cost (Greedy SA): {gsa_cost_mean:.2f} ± {gsa_cost_ci:.2f}")
#         print(f"Mean Iterations (Greedy SA): {gsa_iter_mean:.2f} ± {gsa_iter_ci:.2f}")

#     if len(reluctant_sa_costs):
#         rsa_cost_mean, rsa_cost_ci = mean_ci(reluctant_sa_costs)
#         rsa_iter_mean, rsa_iter_ci = mean_ci(reluctant_sa_iters)
#         print(f"Mean Final Cost (Reluctant SA): {rsa_cost_mean:.2f} ± {rsa_cost_ci:.2f}")
#         print(f"Mean Iterations (Reluctant SA): {rsa_iter_mean:.2f} ± {rsa_iter_ci:.2f}")

def compute_summary_table(cost_data, num_nodes, graph_name):
    greedy_costs = []
    reluctant_costs = []
    greedy_iters = []
    reluctant_iters = []

    for run in cost_data.values():
        try:
            if "cost_data_gr" in run and "cost_data_rr" in run:
                # Get the last element as scalar
                g_final = run["cost_data_gr"][-1]
                r_final = run["cost_data_rr"][-1]

                if isinstance(g_final, (list, np.ndarray)):
                    g_final = g_final[-1]
                if isinstance(r_final, (list, np.ndarray)):
                    r_final = r_final[-1]

                greedy_costs.append(float(g_final))
                reluctant_costs.append(float(r_final))
                greedy_iters.append(len(run["cost_data_gr"]))
                reluctant_iters.append(len(run["cost_data_rr"]))
        except Exception as e:
            print(f"Skipping malformed entry due to error: {e}")

    greedy_costs = np.array(greedy_costs)
    reluctant_costs = np.array(reluctant_costs)
    greedy_iters = np.array(greedy_iters)
    reluctant_iters = np.array(reluctant_iters)

    def mean_ci(arr):
        mean = np.mean(arr)
        ci = 1.96 * np.std(arr, ddof=1) / np.sqrt(len(arr))  # 95% CI
        return mean, ci

    g_mean, g_ci = mean_ci(greedy_costs)
    r_mean, r_ci = mean_ci(reluctant_costs)
    cost_diff = greedy_costs - reluctant_costs
    diff_mean, diff_ci = mean_ci(cost_diff)
    g_iter_mean, g_iter_ci = mean_ci(greedy_iters)
    r_iter_mean, r_iter_ci = mean_ci(reluctant_iters)

    print(f"--- Summary for {graph_name} (n={num_nodes}) ---")
    print(f"Mean Final Cost (Greedy): {g_mean:.2f} ± {g_ci:.2f}")
    print(f"Mean Final Cost (Reluctant): {r_mean:.2f} ± {r_ci:.2f}")
    print(f"Cost Difference (G - R): {diff_mean:.2f} ± {diff_ci:.2f}")
    print(f"Mean Iterations (Greedy): {g_iter_mean:.2f} ± {g_iter_ci:.2f}")
    print(f"Mean Iterations (Reluctant): {r_iter_mean:.2f} ± {r_iter_ci:.2f}")


if __name__ == '__main__':

    file_path = r"C:\Projects\Heuristics for combinatorial optimisation\results\(7500, 20, 2, 0)_results.json"

    with open(file_path, 'r') as f:
        data = json.load(f)

    graph_name = data['graph_name']
    degree = data['degree']
    num_nodes = data['num_nodes']
    color_set_size = data['color_set_size']
    gaussian_mean = data['gaussian_mean']
    gaussian_variance = data['gaussian_variance']
    all_cost_data = data['cost_data']

    # get best final cost for both reluctant and greedy, compare to see global optima
    # best_costs = get_best_final_cost(cost_data=all_cost_data)
    # if best_costs['Greedy']['Best Final Cost'] < best_costs['Reluctant']['Best Final Cost']:
    #     print(f"Greedy found global optima. Final cost: {best_costs['Greedy']['Best Final Cost']} vs {best_costs['Reluctant']['Best Final Cost']}")
    # else:
        # print(f"Reluctant found global optima. Final cost: {best_costs['Reluctant']['Best Final Cost']} vs {best_costs['Greedy']['Best Final Cost']}")
    
    # visualisation: plot cost against iterations for all initial colorings 
    # plot_cost_data(all_cost_data, graph_name, color_set_size, degree, num_nodes, gaussian_mean, gaussian_variance, specific_coloring=None)

    # stats = calculate_greedy_vs_reluctant_stats(data["cost_data"])

    # print(f"Probability that greedy is better: {stats['probability_greedy_better']}")
    # print(f"Average final cost (Greedy): {stats['avg_greedy_final_cost']}")
    # print(f"Average final cost (Reluctant): {stats['avg_reluctant_final_cost']}")
    # print(f"Average cost difference (Greedy - Reluctant): {stats['avg_cost_difference']}")

    # visualisation: plot scatter of final costs against initial colorings
    # plot_final_costs(
    #     cost_data=data["cost_data"],
    #     graph_name=graph_name,
    #     degree=degree,
    #     num_nodes=num_nodes,
    #     color_set_size=color_set_size,
    #     gaussian_mean=gaussian_mean,
    #     gaussian_variance=gaussian_variance
    # )

    # avg_norm_cost_diff = avg_norm_cost_diff(all_cost_data, num_nodes)
    # print(f"Average normalized cost difference: {avg_norm_cost_diff}")

    # visualisation: plot histogram of cost differences for all initial colorings
    # plot_cost_diff_histogram(data["cost_data"], num_nodes, graph_name, num_bins=100, bin_range=(-0.2, 0.2))

    compute_summary_table(all_cost_data, num_nodes, graph_name)