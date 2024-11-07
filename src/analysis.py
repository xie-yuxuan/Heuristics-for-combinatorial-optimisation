import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os

from visualisation import plot_cost_data


if __name__ == '__main__':

    file_path = r"C:\Projects\Heuristics for combinatorial optimisation\Heuristics-for-combinatorial-optimisation\results\expt5_results.json"

    with open(file_path, 'r') as f:
        data = json.load(f)

    graph_name = data['graph_name']
    degree = data['degree']
    num_nodes = data['num_nodes']
    color_set_size = data['color_set_size']
    gaussian_mean = data['gaussian_mean']
    gaussian_variance = data['gaussian_variance']
    
    greedy_better_count = 0
    cost_differences = []
    greedy_final_costs = []
    reluctant_final_costs = []

    for iteration_key, iteration_data in data["cost_data"].items():
        cost_data_g = iteration_data["cost_data_g"]
        cost_data_r = iteration_data["cost_data_r"]

        final_cost_g = cost_data_g[1][-1]  # Last entry in the greedy cost data
        final_cost_r = cost_data_r[1][-1]  # Last entry in the reluctant cost data

        greedy_final_costs.append(final_cost_g)
        reluctant_final_costs.append(final_cost_r)
        
        if final_cost_g < final_cost_r:
            greedy_better_count += 1
        
        cost_difference = final_cost_g - final_cost_r
        cost_differences.append(cost_difference)

    average_cost_difference = np.mean(cost_differences)

    print(f"Probability that greedy is better: {greedy_better_count/len(cost_differences)}")
    print(f"Average cost difference (Greedy - Reluctant): {average_cost_difference}") # positive means greedy converged to higher final cost

    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(greedy_final_costs)), greedy_final_costs, label='Greedy', color='red', alpha=0.6)
    plt.scatter(range(len(reluctant_final_costs)), reluctant_final_costs, label='Reluctant', color='green', alpha=0.6)
    plt.axhline(np.mean(greedy_final_costs), color='red', linestyle='--', label=f'Mean Greedy: {np.mean(greedy_final_costs):.2f}')
    plt.axhline(np.mean(reluctant_final_costs), color='green', linestyle='--', label=f'Mean Reluctant: {np.mean(reluctant_final_costs):.2f}')

    plt.xlabel('Instance')
    plt.ylabel('Final Cost')
    plt.title(f'Greedy and reluctant final costs for multiple instances of {graph_name}')

    experiment_text = f"Degree: {degree}\nNum Nodes: {num_nodes}\nColor Set Size: {color_set_size}\nGaussian Mean: {gaussian_mean}\nGaussian Variance: {gaussian_variance}"
    plt.gca().text(0.95, 0.95, experiment_text, transform=plt.gca().transAxes, fontsize=8, 
               verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5))
    plt.legend()

    plt.show()

    # # plot comparison for particular instance ------------------------------------------

    instance = 0 # change this to see other instances
    cost_data_g = data["cost_data"][f"instance_{instance}"]["cost_data_g"]
    cost_data_r = data["cost_data"][f"instance_{instance}"]["cost_data_r"]

    plot_cost_data( # comparison btw greedy and reluctant results for specified instance
        cost_data_g, len(cost_data_g[0]), cost_data_g[1][-1], 
        cost_data_r, len(cost_data_r[0]), cost_data_r[1][-1],
        graph_name, color_set_size, degree, num_nodes, gaussian_mean, gaussian_variance
        )
    


