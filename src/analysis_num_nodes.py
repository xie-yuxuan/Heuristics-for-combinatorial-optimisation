import os
import json
import numpy as np
import matplotlib.pyplot as plt

'''
Run this program to show plots for graphs of all num_nodes.
'''

def plot_histogram_for_all_num_nodes(results_folder, degree, color_set_size, num_bins=20, bin_range=None):
    '''
    Plot histograms for all graphs of diff num_nodes.
    '''
    # Get a list of all files in the results folder
    result_files = [f for f in os.listdir(results_folder) if f.endswith('_results.json')]
    
    # Initialize a list to store the cost differences for all graphs
    all_cost_differences = []
    labels = []
    
    # Define a color cycle (you can define more colors if necessary)
    color_cycle = plt.cm.tab10.colors  # This gives you 10 distinct colors

    # Find the global minimum and maximum cost differences to set a consistent bin range
    global_min_cost_diff = float('inf')
    global_max_cost_diff = float('-inf')
    
    # First pass: Calculate all cost differences and determine global min/max for bin range
    for i, result_file in enumerate(result_files):
        # Extract graph parameters (num_nodes, degree, color_set_size) from the file name
        file_name_parts = result_file.split('_')[0].strip('()').split(', ')
        num_nodes = int(file_name_parts[0])
        current_degree = int(file_name_parts[1])
        current_color_set_size = int(file_name_parts[2])

        # Check if the file matches the provided degree and color_set_size
        if current_degree == degree and current_color_set_size == color_set_size:
            # Construct the full file path
            file_path = os.path.join(results_folder, result_file)
            
            # Load the JSON data from the result file
            with open(file_path, 'r') as f:
                cost_data = json.load(f)["cost_data"]
            
            # Collect all the cost differences for the current graph
            cost_differences = []
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

                # Update global min/max
                if normalized_cost_diff < global_min_cost_diff:
                    global_min_cost_diff = normalized_cost_diff
                if normalized_cost_diff > global_max_cost_diff:
                    global_max_cost_diff = normalized_cost_diff

                # Append to the list
                cost_differences.append(normalized_cost_diff)

            # Append the cost differences and labels for plotting
            all_cost_differences.append(cost_differences)
            labels.append(f"{num_nodes} nodes")

    # Sort the labels and corresponding cost differences by the number of nodes
    sorted_labels, sorted_cost_differences = zip(*sorted(zip(labels, all_cost_differences), key=lambda x: int(x[0].split()[0])))

    # Set bin range based on global min/max values
    if bin_range is None:
        bin_range = (global_min_cost_diff, global_max_cost_diff)
    
    # Plot the histograms for all graphs on the same axes
    plt.figure(figsize=(10, 6))
    
    # Overlay the histograms for all graphs
    for i, cost_differences in enumerate(sorted_cost_differences):
        plt.hist(cost_differences, bins=num_bins, edgecolor='black', alpha=0.5, 
                 range=bin_range, label=sorted_labels[i], color=color_cycle[i % len(color_cycle)])

    plt.xlabel('Normalized Cost Difference (1/n * (Greedy - Reluctant))')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Normalized Cost Differences (Degree {degree}, Color Set Size {color_set_size})')

    plt.legend(title="Number of Nodes")

    plt.grid(True)

    plt.savefig(f"plots/(all, {degree}, {color_set_size})_norm_cost_diff_hist.png")
    plt.show()

def plot_final_cost_for_all_num_nodes(results_folder, degree, color_set_size):
    '''
    Plot final, avg cost + var, against num_nodes.
    '''
    # Get a list of all files in the results folder
    result_files = [f for f in os.listdir(results_folder) if f.endswith('_results.json')]
    
    # Lists to hold data for plotting
    num_nodes_list = []
    best_final_cost_greedy = []
    best_final_cost_reluctant = []
    avg_final_cost_greedy = []
    avg_final_cost_reluctant = []
    var_final_cost_greedy = []
    var_final_cost_reluctant = []

    # Iterate through all result files
    for result_file in result_files:
        # Extract graph parameters (num_nodes, degree, color_set_size) from the file name
        file_name_parts = result_file.split('_')[0].strip('()').split(', ')
        num_nodes = int(file_name_parts[0])
        current_degree = int(file_name_parts[1])
        current_color_set_size = int(file_name_parts[2])

        # Check if the file matches the provided degree and color_set_size
        if current_degree == degree and current_color_set_size == color_set_size:
            # Construct the full file path
            file_path = os.path.join(results_folder, result_file)
            
            # Load the JSON data from the result file
            with open(file_path, 'r') as f:
                cost_data = json.load(f)["cost_data"]
            
            # Variables to hold the final costs for greedy and reluctant
            final_costs_greedy = []
            final_costs_reluctant = []

            # Iterate through all initial colorings and get final costs
            for initial_coloring_key, iteration_data in cost_data.items():
                cost_data_g = iteration_data["cost_data_g"]
                cost_data_r = iteration_data["cost_data_r"]

                # Get the final costs for both approaches
                final_cost_g = cost_data_g[-1]
                final_cost_r = cost_data_r[-1]

                final_costs_greedy.append(final_cost_g)
                final_costs_reluctant.append(final_cost_r)

            # Calculate best final costs (global optima) for both approaches
            best_final_cost_greedy.append(min(final_costs_greedy))
            best_final_cost_reluctant.append(min(final_costs_reluctant))

            # Calculate average final costs for both approaches
            avg_final_cost_greedy.append(np.mean(final_costs_greedy))
            avg_final_cost_reluctant.append(np.mean(final_costs_reluctant))

            # Calculate variance of final costs for both approaches
            var_final_cost_greedy.append(np.var(final_costs_greedy))
            var_final_cost_reluctant.append(np.var(final_costs_reluctant))

            # Append num_nodes to the list for plotting
            num_nodes_list.append(num_nodes)

    # Sort the data by the number of nodes
    sorted_indices = np.argsort(num_nodes_list)
    num_nodes_list_sorted = np.array(num_nodes_list)[sorted_indices]
    best_final_cost_greedy_sorted = np.array(best_final_cost_greedy)[sorted_indices]
    best_final_cost_reluctant_sorted = np.array(best_final_cost_reluctant)[sorted_indices]
    avg_final_cost_greedy_sorted = np.array(avg_final_cost_greedy)[sorted_indices]
    avg_final_cost_reluctant_sorted = np.array(avg_final_cost_reluctant)[sorted_indices]
    var_final_cost_greedy_sorted = np.array(var_final_cost_greedy)[sorted_indices]
    var_final_cost_reluctant_sorted = np.array(var_final_cost_reluctant)[sorted_indices]

    plt.figure(figsize=(10, 6))

    # Plot best final costs
    plt.plot(num_nodes_list_sorted, best_final_cost_greedy_sorted, label="Best Final Cost (Greedy)", marker='o', color='red', linestyle='-', markersize=8)
    plt.plot(num_nodes_list_sorted, best_final_cost_reluctant_sorted, label="Best Final Cost (Reluctant)", marker='o', color='green', linestyle='-', markersize=8)

    # Plot average final costs
    plt.plot(num_nodes_list_sorted, avg_final_cost_greedy_sorted, label="Avg Final Cost (Greedy)", marker='o', color='red', linestyle='--', markersize=8, alpha=0.3)
    plt.plot(num_nodes_list_sorted, avg_final_cost_reluctant_sorted, label="Avg Final Cost (Reluctant)", marker='o', color='green', linestyle='--', markersize=8, alpha=0.3)

    # Plot variance as shaded area (alpha = 0.5)
    plt.fill_between(num_nodes_list_sorted, avg_final_cost_greedy_sorted - 2*np.sqrt(var_final_cost_greedy_sorted),
                     avg_final_cost_greedy_sorted + 2*np.sqrt(var_final_cost_greedy_sorted), color='red', alpha=0.1, label="Greedy Variance")
    plt.fill_between(num_nodes_list_sorted, avg_final_cost_reluctant_sorted - 2*np.sqrt(var_final_cost_reluctant_sorted),
                     avg_final_cost_reluctant_sorted + 2*np.sqrt(var_final_cost_reluctant_sorted), color='green', alpha=0.1, label="Reluctant Variance")

    plt.xlabel('Number of Nodes')
    plt.ylabel('Cost')
    plt.title(f'Final Cost vs Number of Nodes (Degree {degree}, Color Set Size {color_set_size})')

    plt.legend()

    plt.grid(True)

    plt.savefig(f"plots/(all, {degree}, {color_set_size})_cost.png")
    plt.show()
def plot_norm_final_cost_for_all_num_nodes(results_folder, degree, color_set_size):
    '''
    Plot final, avg cost + var, against num_nodes.
    '''
    # Get a list of all files in the results folder
    result_files = [f for f in os.listdir(results_folder) if f.endswith('_results.json')]
    
    # Lists to hold data for plotting
    num_nodes_list = []
    best_final_cost_greedy = []
    best_final_cost_reluctant = []
    avg_final_cost_greedy = []
    avg_final_cost_reluctant = []
    var_final_cost_greedy = []
    var_final_cost_reluctant = []

    # Iterate through all result files
    for result_file in result_files:
        # Extract graph parameters (num_nodes, degree, color_set_size) from the file name
        file_name_parts = result_file.split('_')[0].strip('()').split(', ')
        num_nodes = int(file_name_parts[0])
        current_degree = int(file_name_parts[1])
        current_color_set_size = int(file_name_parts[2])

        # Check if the file matches the provided degree and color_set_size
        if current_degree == degree and current_color_set_size == color_set_size:
            # Construct the full file path
            file_path = os.path.join(results_folder, result_file)
            
            # Load the JSON data from the result file
            with open(file_path, 'r') as f:
                cost_data = json.load(f)["cost_data"]
            
            # Variables to hold the final costs for greedy and reluctant
            final_costs_greedy = []
            final_costs_reluctant = []

            # Iterate through all initial colorings and get final costs
            for initial_coloring_key, iteration_data in cost_data.items():
                cost_data_g = iteration_data["cost_data_g"]
                cost_data_r = iteration_data["cost_data_r"]

                # Get the final costs for both approaches
                final_cost_g = cost_data_g[-1]
                final_cost_r = cost_data_r[-1]

                final_costs_greedy.append(final_cost_g)
                final_costs_reluctant.append(final_cost_r)

            # Calculate best final costs (global optima) for both approaches
            best_final_cost_greedy.append(min(final_costs_greedy))
            best_final_cost_reluctant.append(min(final_costs_reluctant))

            # Calculate average final costs for both approaches
            avg_final_cost_greedy.append(np.mean(final_costs_greedy))
            avg_final_cost_reluctant.append(np.mean(final_costs_reluctant))

            # Calculate variance of final costs for both approaches
            var_final_cost_greedy.append(np.var(final_costs_greedy))
            var_final_cost_reluctant.append(np.var(final_costs_reluctant))

            # Append num_nodes to the list for plotting
            num_nodes_list.append(num_nodes)

    # Sort the data by the number of nodes
    sorted_indices = np.argsort(num_nodes_list)
    num_nodes_list_sorted = np.array(num_nodes_list)[sorted_indices]
    best_final_cost_greedy_sorted = np.array(best_final_cost_greedy)[sorted_indices]
    best_final_cost_reluctant_sorted = np.array(best_final_cost_reluctant)[sorted_indices]
    avg_final_cost_greedy_sorted = np.array(avg_final_cost_greedy)[sorted_indices]
    avg_final_cost_reluctant_sorted = np.array(avg_final_cost_reluctant)[sorted_indices]
    var_final_cost_greedy_sorted = np.array(var_final_cost_greedy)[sorted_indices]
    var_final_cost_reluctant_sorted = np.array(var_final_cost_reluctant)[sorted_indices]

    # Normalize the costs by dividing by num_nodes
    best_final_cost_greedy_norm = best_final_cost_greedy_sorted / num_nodes_list_sorted
    best_final_cost_reluctant_norm = best_final_cost_reluctant_sorted / num_nodes_list_sorted
    avg_final_cost_greedy_norm = avg_final_cost_greedy_sorted / num_nodes_list_sorted
    avg_final_cost_reluctant_norm = avg_final_cost_reluctant_sorted / num_nodes_list_sorted

    plt.figure(figsize=(10, 6))

    # Plot normalized best final costs
    plt.plot(num_nodes_list_sorted, best_final_cost_greedy_norm, label="Best Final Cost (Greedy)", marker='o', color='red', linestyle='-', markersize=8)
    plt.plot(num_nodes_list_sorted, best_final_cost_reluctant_norm, label="Best Final Cost (Reluctant)", marker='o', color='green', linestyle='-', markersize=8)

    # Plot normalized average final costs
    plt.plot(num_nodes_list_sorted, avg_final_cost_greedy_norm, label="Avg Final Cost (Greedy)", marker='o', color='red', linestyle='--', markersize=8, alpha=0.3)
    plt.plot(num_nodes_list_sorted, avg_final_cost_reluctant_norm, label="Avg Final Cost (Reluctant)", marker='o', color='green', linestyle='--', markersize=8, alpha=0.3)

    # Plot normalized variance as shaded area (alpha = 0.5)
    plt.fill_between(num_nodes_list_sorted, avg_final_cost_greedy_norm - 2*np.sqrt(var_final_cost_greedy_sorted) / num_nodes_list_sorted,
                     avg_final_cost_greedy_norm + 2*np.sqrt(var_final_cost_greedy_sorted) / num_nodes_list_sorted, color='red', alpha=0.1, label="Greedy Variance")
    plt.fill_between(num_nodes_list_sorted, avg_final_cost_reluctant_norm - 2*np.sqrt(var_final_cost_reluctant_sorted) / num_nodes_list_sorted,
                     avg_final_cost_reluctant_norm + 2*np.sqrt(var_final_cost_reluctant_sorted) / num_nodes_list_sorted, color='green', alpha=0.1, label="Reluctant Variance")

    plt.xlabel('Number of Nodes')
    plt.ylabel('Normalized Cost (Cost / Num Nodes)')
    plt.title(f'Normalized Final Cost vs Number of Nodes (Degree {degree}, Color Set Size {color_set_size})')

    plt.legend()

    plt.grid(True)

    plt.savefig(f"plots/(all, {degree}, {color_set_size})_norm_cost.png")
    plt.show()


def plot_iteration_for_all_num_nodes(results_folder, degree, color_set_size):
    '''
    Plot avg iterations + var, num of iter for best case against num_nodes.
    '''
    # Get a list of all files in the results folder
    result_files = [f for f in os.listdir(results_folder) if f.endswith('_results.json')]
    
    # Lists to hold data for plotting
    num_nodes_list = []
    avg_iterations_greedy = []
    avg_iterations_reluctant = []
    avg_iterations_greedy_variance = []
    avg_iterations_reluctant_variance = []
    best_case_iterations_greedy = []
    best_case_iterations_reluctant = []

    # Iterate through all result files
    for result_file in result_files:
        # Extract graph parameters (num_nodes, degree, color_set_size) from the file name
        file_name_parts = result_file.split('_')[0].strip('()').split(', ')
        num_nodes = int(file_name_parts[0])
        current_degree = int(file_name_parts[1])
        current_color_set_size = int(file_name_parts[2])

        # Check if the file matches the provided degree and color_set_size
        if current_degree == degree and current_color_set_size == color_set_size:
            # Construct the full file path
            file_path = os.path.join(results_folder, result_file)
            
            # Load the JSON data from the result file
            with open(file_path, 'r') as f:
                cost_data = json.load(f)["cost_data"]
            
            # Variables to track best-case iterations for greedy and reluctant
            best_iterations_greedy = []
            best_iterations_reluctant = []
            iterations_greedy = []
            iterations_reluctant = []
            best_cost_greedy = float('inf')
            best_cost_reluctant = float('inf')

            # Iterate through all initial colorings to find the best-case and calculate averages
            for initial_coloring_key, iteration_data in cost_data.items():
                cost_data_g = iteration_data["cost_data_g"]
                cost_data_r = iteration_data["cost_data_r"]

                # Get the final costs and iterations for greedy and reluctant
                final_cost_g = cost_data_g[-1]  # Last cost for Greedy
                final_cost_r = cost_data_r[-1]  # Last cost for Reluctant
                iterations_g = len(cost_data_g)
                iterations_r = len(cost_data_r)

                # Track all iterations for calculating the average
                iterations_greedy.append(iterations_g)
                iterations_reluctant.append(iterations_r)

                # Identify the initial coloring that results in the lowest final cost for Greedy
                if final_cost_g < best_cost_greedy:
                    best_cost_greedy = final_cost_g
                    best_iterations_greedy = [iterations_g]

                # Identify the initial coloring that results in the lowest final cost for Reluctant
                if final_cost_r < best_cost_reluctant:
                    best_cost_reluctant = final_cost_r
                    best_iterations_reluctant = [iterations_r]

            # Ensure we don't try to calculate min on empty lists
            if best_iterations_greedy:
                best_case_iterations_greedy.append(np.min(best_iterations_greedy))
            else:
                best_case_iterations_greedy.append(np.nan)  # Use NaN or a large number as a placeholder

            if best_iterations_reluctant:
                best_case_iterations_reluctant.append(np.min(best_iterations_reluctant))
            else:
                best_case_iterations_reluctant.append(np.nan)  # Use NaN or a large number as a placeholder

            # Calculate average and variance of iterations for both greedy and reluctant
            avg_iterations_greedy.append(np.mean(iterations_greedy))
            avg_iterations_reluctant.append(np.mean(iterations_reluctant))
            avg_iterations_greedy_variance.append(np.var(iterations_greedy))
            avg_iterations_reluctant_variance.append(np.var(iterations_reluctant))

            # Append num_nodes to the list for plotting
            num_nodes_list.append(num_nodes)

    # Sort the data by the number of nodes
    sorted_indices = np.argsort(num_nodes_list)
    num_nodes_list_sorted = np.array(num_nodes_list)[sorted_indices]
    avg_iterations_greedy_sorted = np.array(avg_iterations_greedy)[sorted_indices]
    avg_iterations_reluctant_sorted = np.array(avg_iterations_reluctant)[sorted_indices]
    avg_iterations_greedy_variance_sorted = np.array(avg_iterations_greedy_variance)[sorted_indices]
    avg_iterations_reluctant_variance_sorted = np.array(avg_iterations_reluctant_variance)[sorted_indices]
    best_case_iterations_greedy_sorted = np.array(best_case_iterations_greedy)[sorted_indices]
    best_case_iterations_reluctant_sorted = np.array(best_case_iterations_reluctant)[sorted_indices]

    plt.figure(figsize=(10, 6))

    # Plot average iterations for Greedy and Reluctant with variance
    plt.plot(num_nodes_list_sorted, avg_iterations_greedy_sorted, label="Average Iterations (Greedy)", marker='o', color='red', linestyle='--', markersize=8,alpha=0.3)
    plt.plot(num_nodes_list_sorted, avg_iterations_reluctant_sorted, label="Average Iterations (Reluctant)", marker='o', color='green', linestyle='--', markersize=8, alpha=0.3)

    # Plot the best case iterations for Greedy and Reluctant (iterations for lowest final cost)
    plt.plot(num_nodes_list_sorted, best_case_iterations_greedy_sorted, label="Best Case Iterations (Greedy)", marker='x', color='red', linestyle='-', markersize=8)
    plt.plot(num_nodes_list_sorted, best_case_iterations_reluctant_sorted, label="Best Case Iterations (Reluctant)", marker='x', color='green', linestyle='-', markersize=8)

    # Add error bars for variance
    plt.fill_between(num_nodes_list_sorted, avg_iterations_greedy_sorted - np.sqrt(avg_iterations_greedy_variance_sorted), avg_iterations_greedy_sorted + np.sqrt(avg_iterations_greedy_variance_sorted), alpha=0.1, color='red')
    plt.fill_between(num_nodes_list_sorted, avg_iterations_reluctant_sorted - np.sqrt(avg_iterations_reluctant_variance_sorted), avg_iterations_reluctant_sorted + np.sqrt(avg_iterations_reluctant_variance_sorted), alpha=0.1, color='green')

    plt.xlabel('Number of Nodes')
    plt.ylabel('Number of Iterations to Convergence')
    plt.title(f'Average and Best Case Iterations to Convergence vs Number of Nodes (Degree {degree}, Color Set Size {color_set_size})')

    plt.legend()

    plt.grid(True)

    plt.savefig(f"plots/(all, {degree}, {color_set_size})_iter.png")
    plt.show()


if __name__ == "__main__":
    results_folder = r'C:\Projects\Heuristics for combinatorial optimisation\results'

    # set degree and color_set_size
    degree = 10
    color_set_size = 8

    plot_final_cost_for_all_num_nodes(results_folder, degree=degree, color_set_size=color_set_size)
    plot_norm_final_cost_for_all_num_nodes(results_folder, degree=degree, color_set_size=color_set_size)
    plot_iteration_for_all_num_nodes(results_folder, degree, color_set_size)
    plot_histogram_for_all_num_nodes(results_folder, degree=degree, color_set_size=color_set_size, num_bins=100, bin_range=None)

    