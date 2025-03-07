import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os

from visualisation import plot_cost_data, plot_final_costs, plot_cost_diff_histogram

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
        iterations_fg, costs_fg = value["cost_data_g"]
        iterations_fr, costs_fr = value["cost_data_r"]

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

        final_cost_g = cost_data_g[1][-1]  # Last entry in the greedy cost data
        final_cost_r = cost_data_r[1][-1]  # Last entry in the reluctant cost data

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

        final_cost_g = cost_data_g[1][-1]  # Last entry in the greedy cost data
        final_cost_r = cost_data_r[1][-1]  # Last entry in the reluctant cost data

        # Calculate the cost difference
        cost_difference = final_cost_g - final_cost_r
        cost_differences.append(cost_difference)

    # Calculate the average and standard deviation of the cost differences
    avg_cost_difference = np.mean(cost_differences)
    std_cost_difference = np.std(cost_differences)

    return avg_cost_difference / num_nodes, std_cost_difference / num_nodes

def plot_cost_data(cost_data, graph_name, color_set_size, degree, num_nodes, gaussian_mean, gaussian_variance, specific_coloring=None):
    '''
    Plot cost against iterations for one graph.
    '''
    plt.figure(figsize=(10, 6))

    min_final_cost_g = float('inf')
    min_final_cost_r = float('inf')
    min_iterations_fg = None
    min_iterations_fr = None

    # Iterate over each initial coloring in cost_data
    for i, (key, value) in enumerate(cost_data.items()):
        # Skip other colorings if a specific coloring is selected
        if specific_coloring is not None and i != specific_coloring:
            continue

        # Extract data for the greedy and reluctant algorithms
        iterations_fg, costs_fg = value["cost_data_g"]
        iterations_fr, costs_fr = value["cost_data_r"]

        # Final cost and total iterations for this coloring
        total_iterations_fg = iterations_fg[-1]
        final_cost_fg = costs_fg[-1]
        total_iterations_fr = iterations_fr[-1]
        final_cost_fr = costs_fr[-1]

        # Plot "Greedy" (fg) with transparency for multiple colorings
        plt.scatter(iterations_fg, costs_fg, color="red", alpha=0.6)
        plt.scatter(total_iterations_fg, final_cost_fg, color="red", s=50, alpha=0.6)

        # Plot "Reluctant" (fr) with transparency for multiple colorings
        plt.scatter(iterations_fr, costs_fr, color="green", alpha=0.6)
        plt.scatter(total_iterations_fr, final_cost_fr, color="green", s=50, alpha=0.6)

        # Update minimum final cost and iterations for Greedy
        if final_cost_fg < min_final_cost_g:
            min_final_cost_g = final_cost_fg
            min_iterations_fg = total_iterations_fg

        # Update minimum final cost and iterations for Reluctant
        if final_cost_fr < min_final_cost_r:
            min_final_cost_r = final_cost_fr
            min_iterations_fr = total_iterations_fr

        # Add dashed line to show the gap if fg converges earlier than fr
        if total_iterations_fg < total_iterations_fr:
            plt.hlines(final_cost_fg, total_iterations_fg, total_iterations_fr, colors="purple", linestyles="dashed", alpha=0.3)

    # Add annotations for the minimum final cost for Greedy and Reluctant
    if min_iterations_fg is not None:
        plt.annotate(f"Iter: {min_iterations_fg}\nCost: {min_final_cost_g}", 
                     (min_iterations_fg, min_final_cost_g), textcoords="offset points", 
                     xytext=(10, -15), ha='center', color="red", fontsize=8)

    if min_iterations_fr is not None:
        plt.annotate(f"Iter: {min_iterations_fr}\nCost: {min_final_cost_r}", 
                     (min_iterations_fr, min_final_cost_r), textcoords="offset points", 
                     xytext=(10, -15), ha='center', color="green", fontsize=8)

    plt.plot([], [], color="red", label="Greedy")
    plt.plot([], [], color="green", label="Reluctant")
    plt.legend(loc="upper right")
    
    param_text = (f"Color Set Size: {color_set_size}\n"
                  f"Degree: {degree}\n"
                  f"Number of Nodes: {num_nodes}\n"
                  f"Gaussian Mean: {gaussian_mean}\n"
                  f"Gaussian Variance: {gaussian_variance}")
    plt.gcf().text(0.75, 0.65, param_text, fontsize=8, bbox=dict(facecolor='white', alpha=0.5))

    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title(f"Cost vs Iterations for Greedy and Reluctant Algorithms on {graph_name}")
    plt.grid()

    plt.savefig(f"plots/{graph_name}_cost.png")

    plt.show()




def plot_final_costs(cost_data, graph_name, degree, num_nodes, color_set_size, gaussian_mean, gaussian_variance):
    '''
    Plot a scatter plot of final costs against initial coloring. 
    Show average final cost.
    '''
    greedy_final_costs = []
    reluctant_final_costs = []

    # Extract the final cost for each initial coloring for both greedy and reluctant algorithms
    for initial_coloring_key, iteration_data in cost_data.items():
        cost_data_g = iteration_data["cost_data_g"]
        cost_data_r = iteration_data["cost_data_r"]

        final_cost_g = cost_data_g[1][-1]  # Last entry in the greedy cost data
        final_cost_r = cost_data_r[1][-1]  # Last entry in the reluctant cost data

        greedy_final_costs.append(final_cost_g)
        reluctant_final_costs.append(final_cost_r)

    plt.figure(figsize=(10, 6))

    # Scatter plot for greedy and reluctant final costs
    plt.scatter(range(len(greedy_final_costs)), greedy_final_costs, label='Greedy', color='red', alpha=0.6)
    plt.scatter(range(len(reluctant_final_costs)), reluctant_final_costs, label='Reluctant', color='green', alpha=0.6)

    # Mean lines for greedy and reluctant
    plt.axhline(np.mean(greedy_final_costs), color='red', linestyle='--', label=f'Mean Greedy: {np.mean(greedy_final_costs)}')
    plt.axhline(np.mean(reluctant_final_costs), color='green', linestyle='--', label=f'Mean Reluctant: {np.mean(reluctant_final_costs)}')

    plt.xlabel('Initial Coloring Index')
    plt.ylabel('Final Cost')
    plt.title(f'Greedy and Reluctant Final Costs for Multiple Initial Colorings of {graph_name}')

    experiment_text = (f"Degree: {degree}\nNum Nodes: {num_nodes}\nColor Set Size: {color_set_size}\n"
                       f"Gaussian Mean: {gaussian_mean}\nGaussian Variance: {gaussian_variance}")
    plt.gca().text(0.95, 0.8, experiment_text, transform=plt.gca().transAxes, fontsize=8, 
                   verticalalignment='bottom', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5))

    plt.legend(loc='upper left')

    plt.savefig(f"plots/{graph_name}_scatter.png")

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
        final_cost_g = cost_data_g[1][-1]
        final_cost_r = cost_data_r[1][-1]

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
    
    plt.xlabel('Normalized Cost Difference (1/n * (Greedy - Reluctant))')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Normalized Cost Differences for {graph_name}')
    
    plt.grid(True)

    plt.savefig(f"plots/{graph_name}_hist.png")

    plt.show()



if __name__ == '__main__':

    file_path = r"C:\Projects\Heuristics for combinatorial optimisation\Heuristics-for-combinatorial-optimisation\results\(1000, 20, 8)_results.json"

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
    best_costs = get_best_final_cost(cost_data=all_cost_data)
    if best_costs['Greedy']['Best Final Cost'] < best_costs['Reluctant']['Best Final Cost']:
        print(f"Greedy found global optima. Final cost: {best_costs['Greedy']['Best Final Cost']} vs {best_costs['Reluctant']['Best Final Cost']}")
    else:
        print(f"Reluctant found global optima. Final cost: {best_costs['Reluctant']['Best Final Cost']} vs {best_costs['Greedy']['Best Final Cost']}")
    
    # visualisation: plot cost against iterations for all initial colorings 
    # plot_cost_data(all_cost_data, graph_name, color_set_size, degree, num_nodes, gaussian_mean, gaussian_variance, specific_coloring=None)

    stats = calculate_greedy_vs_reluctant_stats(data["cost_data"])

    print(f"Probability that greedy is better: {stats['probability_greedy_better']}")
    print(f"Average final cost (Greedy): {stats['avg_greedy_final_cost']}")
    print(f"Average final cost (Reluctant): {stats['avg_reluctant_final_cost']}")
    print(f"Average cost difference (Greedy - Reluctant): {stats['avg_cost_difference']}")

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

    avg_norm_cost_diff = avg_norm_cost_diff(all_cost_data, num_nodes)
    print(f"Average normalized cost difference: {avg_norm_cost_diff}")

    # visualisation: plot histogram of cost differences for all initial colorings
    plot_cost_diff_histogram(data["cost_data"], num_nodes, graph_name, num_bins=100, bin_range=(-0.2, 0.2))

