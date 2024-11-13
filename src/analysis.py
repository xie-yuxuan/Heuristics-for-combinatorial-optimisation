import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os

from visualisation import plot_cost_data, plot_final_costs, plot_cost_diff_histogram

def get_best_final_cost(cost_data):
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

    # Return results as a dictionary
    return {
        "Greedy": {"Best Final Cost": min_final_cost_g, "Iterations": min_iterations_fg},
        "Reluctant": {"Best Final Cost": min_final_cost_r, "Iterations": min_iterations_fr}
    }


def calculate_greedy_vs_reluctant_stats(cost_data):
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

    # Return the results
    return {
        "probability_greedy_better": probability_greedy_better,
        "avg_cost_difference": avg_cost_difference,
        "avg_greedy_final_cost": avg_greedy_final_cost,
        "avg_reluctant_final_cost": avg_reluctant_final_cost
    }

if __name__ == '__main__':

    file_path = r"C:\Projects\Heuristics for combinatorial optimisation\Heuristics-for-combinatorial-optimisation\results\(5000, 20)_results.json"

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
    
    # visualisation: plot cost against iterations for all colorings 
    plot_cost_data(all_cost_data, graph_name, color_set_size, degree, num_nodes, gaussian_mean, gaussian_variance, specific_coloring=None)

    stats = calculate_greedy_vs_reluctant_stats(data["cost_data"])

    print(f"Probability that greedy is better: {stats['probability_greedy_better']}")
    print(f"Average final cost (Greedy): {stats['avg_greedy_final_cost']}")
    print(f"Average final cost (Reluctant): {stats['avg_reluctant_final_cost']}")
    print(f"Average cost difference (Greedy - Reluctant): {stats['avg_cost_difference']}")

    # visualisation

    plot_final_costs(
        cost_data=data["cost_data"],
        graph_name=graph_name,
        degree=degree,
        num_nodes=num_nodes,
        color_set_size=color_set_size,
        gaussian_mean=gaussian_mean,
        gaussian_variance=gaussian_variance
    )

    plot_cost_diff_histogram(data["cost_data"], num_nodes, graph_name)


    "hi github"
