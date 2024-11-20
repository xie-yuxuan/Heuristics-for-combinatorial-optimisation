from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from collections import defaultdict
from scipy.interpolate import griddata
import pyvista as pv

from analysis_single_graph import avg_norm_cost_diff

def plot_3d_final_costs(results_folder, filter_num_nodes, filter_color_set_size):
    """
    Plot a 3D graph of lowest final costs when no filter is specified, or a 2D graph when filtered by num_nodes or color_set_size.
    """
    import os
    import json
    import numpy as np
    import matplotlib.pyplot as plt

    # Degree-specific colors
    degree_colors = {
        20: 'red',
        10: 'green',
        5: 'blue',
        2: 'orange'
    }

    # Collect data from files
    data_points = []
    result_files = [f for f in os.listdir(results_folder) if f.endswith('_results.json')]

    for result_file in result_files:
        # Extract parameters (num_nodes, degree, color_set_size) from filename
        file_name_parts = result_file.split('_')[0].strip('()').split(', ')
        num_nodes = int(file_name_parts[0])
        degree = int(file_name_parts[1])
        color_set_size = int(file_name_parts[2])

        # Check filters
        if filter_num_nodes is not None and num_nodes != filter_num_nodes:
            continue
        if filter_color_set_size is not None and color_set_size != filter_color_set_size:
            continue

        # Load data from the file
        file_path = os.path.join(results_folder, result_file)
        with open(file_path, 'r') as f:
            cost_data = json.load(f)["cost_data"]

        final_costs_greedy = []
        final_costs_reluctant = []

        for initial_coloring_key, iteration_data in cost_data.items():
            cost_data_g = iteration_data["cost_data_g"]
            cost_data_r = iteration_data["cost_data_r"]

            final_cost_g = cost_data_g[1][-1]
            final_cost_r = cost_data_r[1][-1]

            final_costs_greedy.append(final_cost_g)
            final_costs_reluctant.append(final_cost_r)

        # Store data points (lowest final costs for each approach)
        min_final_cost_greedy = np.min(final_costs_greedy)
        min_final_cost_reluctant = np.min(final_costs_reluctant)
        data_points.append((num_nodes, degree, color_set_size, min_final_cost_greedy, min_final_cost_reluctant))

    # If no data points match the filter, exit
    if not data_points:
        print("No data points match the specified filters.")
        return

    # Separate data for plotting
    xs = np.array([d[0] for d in data_points])  # num_nodes
    ys = np.array([d[2] for d in data_points])  # color_set_size
    degrees = np.array([d[1] for d in data_points])  # degree
    zs_greedy = np.array([d[3] for d in data_points])  # final costs (greedy)
    zs_reluctant = np.array([d[4] for d in data_points])  # final costs (reluctant)

    # Check if a 2D plot is needed
    if filter_num_nodes is not None or filter_color_set_size is not None:
        # Determine x-axis (color_set_size if num_nodes is fixed, otherwise num_nodes)
        if filter_num_nodes is not None:
            x_axis = ys
            x_label = "Color Set Size"
        else:
            x_axis = xs
            x_label = "Number of Nodes"

        # Create 2D plot
        plt.figure(figsize=(10, 6))
        for degree, color in degree_colors.items():
            # Filter points for the current degree
            mask = (degrees == degree)
            if not mask.any():
                continue

            x_points = x_axis[mask]
            y_greedy = zs_greedy[mask]
            y_reluctant = zs_reluctant[mask]

            # Sort the data by the number of nodes (x_points)
            sorted_indices = np.argsort(x_points)
            x_points_sorted = x_points[sorted_indices]
            y_greedy_sorted = y_greedy[sorted_indices]
            y_reluctant_sorted = y_reluctant[sorted_indices]

            # Plot points
            plt.plot(x_points_sorted, y_greedy_sorted, color=color, label=f"Greedy (Degree {degree})", marker='o', linestyle='-', markersize=8)
            plt.plot(x_points_sorted, y_reluctant_sorted, color=color, label=f"Reluctant (Degree {degree})", marker='x', linestyle='--', markersize=8)

        plt.xlabel(x_label)
        plt.ylabel("Lowest Final Cost")
        if filter_num_nodes is not None:
            plt.title(f"Plot of Lowest Final Costs vs. {x_label} for num_node = {filter_num_nodes}")
        else:
            plt.title(f"Plot of Lowest Final Costs vs. {x_label} for color_set_size = {filter_color_set_size}")
        plt.legend(loc='upper right', fontsize='small')
        plt.grid(True)
        plt.savefig(f"plots/({filter_num_nodes}, all, {filter_color_set_size})_lowest_cost.png")
        plt.show()
    else:
        # Create 3D plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot data for each degree
        for degree, color in degree_colors.items():
            # Filter points for the current degree
            mask = (degrees == degree)
            if not mask.any():
                continue  # Skip if no points for this degree

            xs_degree = xs[mask]
            ys_degree = ys[mask]
            zs_greedy_degree = zs_greedy[mask]
            zs_reluctant_degree = zs_reluctant[mask]

            # Plot points for greedy and reluctant
            ax.scatter(xs_degree, ys_degree, zs_greedy_degree, color=color, label=f"Greedy (Degree {degree})", marker='o')
            ax.scatter(xs_degree, ys_degree, zs_reluctant_degree, color=color, label=f"Reluctant (Degree {degree})", marker='x')

        # Set axis labels
        ax.set_xlabel('Number of Nodes')
        ax.set_ylabel('Color Set Size')
        ax.set_zlabel('Lowest Final Cost')
        ax.set_title('3D Plot of Lowest Final Costs by Degree')
        ax.legend(loc='upper right', fontsize='small')
        plt.savefig(f"plots/(all, all, all)_lowest_cost.png")
        plt.show()


def plot_3d_normalized_costs(results_folder, filter_num_nodes, filter_color_set_size):
    """
    Plot a 3D graph of the best normalized final costs when no filter is specified, 
    or a 2D graph when filtered by num_nodes or color_set_size.
    Normalized final cost is calculated as final_cost / num_nodes.
    """
    import os
    import json
    import numpy as np
    import matplotlib.pyplot as plt

    # Degree-specific colors
    degree_colors = {
        20: 'red',
        10: 'green',
        5: 'blue',
        2: 'orange'
    }

    # Collect data from files
    data_points = []
    result_files = [f for f in os.listdir(results_folder) if f.endswith('_results.json')]

    for result_file in result_files:
        # Extract parameters (num_nodes, degree, color_set_size) from filename
        file_name_parts = result_file.split('_')[0].strip('()').split(', ')
        num_nodes = int(file_name_parts[0])
        degree = int(file_name_parts[1])
        color_set_size = int(file_name_parts[2])

        # Check filters
        if filter_num_nodes is not None and num_nodes != filter_num_nodes:
            continue
        if filter_color_set_size is not None and color_set_size != filter_color_set_size:
            continue

        # Load data from the file
        file_path = os.path.join(results_folder, result_file)
        with open(file_path, 'r') as f:
            cost_data = json.load(f)["cost_data"]

        final_costs_greedy = []
        final_costs_reluctant = []

        for initial_coloring_key, iteration_data in cost_data.items():
            cost_data_g = iteration_data["cost_data_g"]
            cost_data_r = iteration_data["cost_data_r"]

            final_cost_g = cost_data_g[1][-1]
            final_cost_r = cost_data_r[1][-1]

            final_costs_greedy.append(final_cost_g)
            final_costs_reluctant.append(final_cost_r)

        # Store data points (best normalized final costs for each approach)
        best_final_cost_greedy = np.min(final_costs_greedy) / num_nodes
        best_final_cost_reluctant = np.min(final_costs_reluctant) / num_nodes
        data_points.append((num_nodes, degree, color_set_size, best_final_cost_greedy, best_final_cost_reluctant))

    # If no data points match the filter, exit
    if not data_points:
        print("No data points match the specified filters.")
        return

    # Separate data for plotting
    xs = np.array([d[0] for d in data_points])  # num_nodes
    ys = np.array([d[2] for d in data_points])  # color_set_size
    degrees = np.array([d[1] for d in data_points])  # degree
    zs_greedy = np.array([d[3] for d in data_points])  # best normalized costs (greedy)
    zs_reluctant = np.array([d[4] for d in data_points])  # best normalized costs (reluctant)

    # Check if a 2D plot is needed
    if filter_num_nodes is not None or filter_color_set_size is not None:
        # Determine x-axis (color_set_size if num_nodes is fixed, otherwise num_nodes)
        if filter_num_nodes is not None:
            x_axis = ys
            x_label = "Color Set Size"
        else:
            x_axis = xs
            x_label = "Number of Nodes"

        # Create 2D plot
        plt.figure(figsize=(10, 6))
        for degree, color in degree_colors.items():
            # Filter points for the current degree
            mask = (degrees == degree)
            if not mask.any():
                continue

            x_points = x_axis[mask]
            y_greedy = zs_greedy[mask]
            y_reluctant = zs_reluctant[mask]

            # Sort the data by the number of nodes (x_points)
            sorted_indices = np.argsort(x_points)
            x_points_sorted = x_points[sorted_indices]
            y_greedy_sorted = y_greedy[sorted_indices]
            y_reluctant_sorted = y_reluctant[sorted_indices]

            # Plot points
            plt.plot(x_points_sorted, y_greedy_sorted, color=color, label=f"Greedy (Degree {degree})", marker='o', linestyle='-', markersize=8)
            plt.plot(x_points_sorted, y_reluctant_sorted, color=color, label=f"Reluctant (Degree {degree})", marker='x', linestyle='--', markersize=8)

        plt.xlabel(x_label)
        plt.ylabel("Best Normalized Final Cost")
        if filter_num_nodes is not None:
            plt.title(f"Plot of Best Normalized Costs vs. {x_label} for num_node = {filter_num_nodes}")
        else:
            plt.title(f"Plot of Best Normalized Costs vs. {x_label} for color_set_size = {filter_color_set_size}")
        plt.legend(loc='upper right', fontsize='small')
        plt.grid(True)
        plt.savefig(f"plots/({filter_num_nodes}, all, {filter_color_set_size})_best_norm_cost.png")
        plt.show()
    else:
        # Create 3D plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot data for each degree
        for degree, color in degree_colors.items():
            # Filter points for the current degree
            mask = (degrees == degree)
            if not mask.any():
                continue  # Skip if no points for this degree

            xs_degree = xs[mask]
            ys_degree = ys[mask]
            zs_greedy_degree = zs_greedy[mask]
            zs_reluctant_degree = zs_reluctant[mask]

            # Plot points for greedy and reluctant
            ax.scatter(xs_degree, ys_degree, zs_greedy_degree, color=color, label=f"Greedy (Degree {degree})", marker='o')
            ax.scatter(xs_degree, ys_degree, zs_reluctant_degree, color=color, label=f"Reluctant (Degree {degree})", marker='x')

        # Set axis labels
        ax.set_xlabel('Number of Nodes')
        ax.set_ylabel('Color Set Size')
        ax.set_zlabel('Best Normalized Final Cost')
        ax.set_title('3D Plot of Best Normalized Final Costs by Degree')
        ax.legend(loc='upper right', fontsize='small')
        plt.savefig(f"plots/(all, all, all)_best_norm_cost.png")
        plt.show()























def plot_avg_norm_cost_diff(results_folder, num_nodes):
    '''
    Plot average normalized cost difference vs color set size for a fixed num_nodes
    and overlay results for different degrees, including ±2 standard deviation with degree-specific colors.
    '''
    # Get a list of all files in the results folder
    result_files = [f for f in os.listdir(results_folder) if f.endswith('_results.json')]
    
    # Dictionary to hold color set sizes for each degree
    degree_color_set_size_data = {}
    degree_colors = {}  # Dictionary to store unique color for each degree

    # Iterate through all result files
    for result_file in result_files:
        # Extract graph parameters (num_nodes, degree, color_set_size) from the file name
        file_name_parts = result_file.split('_')[0].strip('()').split(', ')
        current_num_nodes = int(file_name_parts[0])
        current_degree = int(file_name_parts[1])
        current_color_set_size = int(file_name_parts[2])

        # Check if the file matches the provided num_nodes
        if current_num_nodes == num_nodes:
            # Construct the full file path
            file_path = os.path.join(results_folder, result_file)
            
            # Load the JSON data from the result file
            with open(file_path, 'r') as f:
                cost_data = json.load(f)["cost_data"]
            
            # Calculate the average normalized cost difference and standard deviation for this color set size
            avg_norm_cost_diff_value, std_norm_cost_diff = avg_norm_cost_diff(cost_data, num_nodes)  # Corrected usage
            
            # Add the data to the dictionary
            if current_degree not in degree_color_set_size_data:
                degree_color_set_size_data[current_degree] = {'color_set_size': [], 'avg_norm_cost_diff': [], 'std_norm_cost_diff': []}
            
            degree_color_set_size_data[current_degree]['color_set_size'].append(current_color_set_size)
            degree_color_set_size_data[current_degree]['avg_norm_cost_diff'].append(avg_norm_cost_diff_value)
            degree_color_set_size_data[current_degree]['std_norm_cost_diff'].append(std_norm_cost_diff)
            
            # Assign a unique color to each degree if not already assigned
            if current_degree not in degree_colors:
                degree_colors[current_degree] = plt.cm.tab10(len(degree_colors) % 10)  # Use a color map to cycle colors
    
    # Plotting the average normalized cost difference for each degree
    plt.figure(figsize=(10, 6))

    for degree, data in degree_color_set_size_data.items():
        color_set_size_sorted = np.array(data['color_set_size'])
        avg_norm_cost_diff_sorted = np.array(data['avg_norm_cost_diff'])
        std_norm_cost_diff_sorted = np.array(data['std_norm_cost_diff'])

        # Sort the data by color set size
        sorted_indices = np.argsort(color_set_size_sorted)
        color_set_size_sorted = color_set_size_sorted[sorted_indices]
        avg_norm_cost_diff_sorted = avg_norm_cost_diff_sorted[sorted_indices]
        std_norm_cost_diff_sorted = std_norm_cost_diff_sorted[sorted_indices]

        # Get the color for the current degree
        degree_color = degree_colors[degree]

        # Plot the data for the current degree
        plt.plot(color_set_size_sorted, avg_norm_cost_diff_sorted, label=f'Degree {degree}', marker='o', color=degree_color, linestyle='-', markersize=8)

        # Plot the shaded area for ± 2 standard deviations with the same color as the line
        plt.fill_between(color_set_size_sorted,
                         avg_norm_cost_diff_sorted - 2 * std_norm_cost_diff_sorted,
                         avg_norm_cost_diff_sorted + 2 * std_norm_cost_diff_sorted,
                         color=degree_color, alpha=0.2)

    plt.xlabel('Color Set Size')
    plt.ylabel('Average Normalized Cost Difference')
    plt.title(f'Average Normalized Cost Difference vs Color Set Size (Num Nodes {num_nodes})')

    plt.legend()

    plt.grid(True)

    plt.savefig(f"plots/{num_nodes}_avg_norm_cost_diff_all_degrees_with_std.png")
    plt.show()


def plot_3d_avg_norm_cost_diff(results_folder, num_nodes):
    '''
    Plot 3D scatter of average normalized cost diff with interpolation, color scale, and save the plot.
    '''
    # Get a list of all files in the results folder
    result_files = [f for f in os.listdir(results_folder) if f.endswith('_results.json')]
    
    # Data containers
    data_points = []

    # Iterate through all result files
    for result_file in result_files:
        # Extract graph parameters (num_nodes, degree, color_set_size) from the file name
        file_name_parts = result_file.split('_')[0].strip('()').split(', ')
        current_num_nodes = int(file_name_parts[0])
        current_degree = int(file_name_parts[1])
        current_color_set_size = int(file_name_parts[2])

        # Only process the files for the specified num_nodes
        if current_num_nodes == num_nodes:
            # Construct the full file path
            file_path = os.path.join(results_folder, result_file)
            
            # Load the JSON data from the result file
            with open(file_path, 'r') as f:
                cost_data = json.load(f)["cost_data"]
            
            # Calculate the average normalized cost difference for this color set size and degree
            avg_norm_cost_diff_value, _ = avg_norm_cost_diff(cost_data, num_nodes)
            
            # Store the data for plotting
            data_points.append([current_color_set_size, current_degree, avg_norm_cost_diff_value])

    # Convert data points to numpy array for easier manipulation
    data_points = np.array(data_points)
    x_vals = data_points[:, 0]  # color set size
    y_vals = data_points[:, 1]  # degree
    z_vals = data_points[:, 2]  # avg normalized cost diff

    # 3D Plotting
    fig = plt.figure(figsize=(10, 8))
    
    # 3D Plot
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(x_vals, y_vals, z_vals, c=z_vals, cmap='RdYlGn', marker='o', label='Data points')

    # Create a grid for interpolation with reduced resolution
    grid_x, grid_y = np.meshgrid(np.linspace(min(x_vals), max(x_vals), 50),  # Reduced grid resolution
                                 np.linspace(min(y_vals), max(y_vals), 50))  # Reduced grid resolution

    # Perform interpolation to fit a plane
    grid_z = griddata((x_vals, y_vals), z_vals, (grid_x, grid_y), method='linear')

    # Plot the interpolated surface
    ax.plot_surface(grid_x, grid_y, grid_z, cmap='RdYlGn', alpha=0.5)

    # Add faint X, Y planes
    ax.plot_surface(grid_x, grid_y, np.zeros_like(grid_z), color='lightgray', alpha=0.3, rstride=100, cstride=100)
    
    ax.set_xlabel('Color Set Size')
    ax.set_ylabel('Degree')
    ax.set_zlabel('Average Normalized Cost Difference')
    ax.set_title(f'3D Plot: Avg Normalized Cost Diff for Num Nodes {num_nodes}')
    # ax.legend()

    # Add color scale
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Avg Normalized Cost Difference')
    cbar.set_ticks(np.linspace(min(z_vals), max(z_vals), 5))  # Adjust color scale ticks

    plt.tight_layout()
    plt.savefig(f"plots/{num_nodes}_3d_avg_norm_cost_diff.png")
    plt.show()



def plot_2d_avg_norm_cost_diff(results_folder, num_nodes):
    '''
    Plot 2D scatter plot of color set size vs degree with average normalized cost difference.
    '''
    # Get a list of all files in the results folder
    result_files = [f for f in os.listdir(results_folder) if f.endswith('_results.json')]

    # Data containers
    data_points = []

    # Iterate through all result files
    for result_file in result_files:
        # Extract graph parameters (num_nodes, degree, color_set_size) from the file name
        file_name_parts = result_file.split('_')[0].strip('()').split(', ')
        current_num_nodes = int(file_name_parts[0])
        current_degree = int(file_name_parts[1])
        current_color_set_size = int(file_name_parts[2])

        # Only process the files for the specified num_nodes
        if current_num_nodes == num_nodes:
            # Construct the full file path
            file_path = os.path.join(results_folder, result_file)
            
            # Load the JSON data from the result file
            with open(file_path, 'r') as f:
                cost_data = json.load(f)["cost_data"]
            
            # Calculate the average normalized cost difference for this color set size and degree
            avg_norm_cost_diff_value, _ = avg_norm_cost_diff(cost_data, num_nodes)
            
            # Store the data for plotting
            data_points.append([current_degree, current_color_set_size, avg_norm_cost_diff_value])

    # Convert data points to numpy array for easier manipulation
    data_points = np.array(data_points)
    x_vals = data_points[:, 0]  # degree
    y_vals = data_points[:, 1]  # color set size
    z_vals = data_points[:, 2]  # avg normalized cost diff

    # 2D Scatter Plot
    plt.figure(figsize=(10, 8))
    
    scatter = plt.scatter(x_vals, y_vals, c=z_vals, cmap='RdYlGn', marker='o', s=100)

    plt.xlabel('Degree')
    plt.ylabel('Color Set Size')
    plt.title(f'2D Scatter: Avg Normalized Cost Diff (Num Nodes {num_nodes})')

    # Add color scale
    cbar = plt.colorbar(scatter)
    cbar.set_label('Avg Normalized Cost Difference')
    cbar.set_ticks(np.linspace(min(z_vals), max(z_vals), 5))  # Adjust color scale ticks

    # Save the plot to a file
    plt.tight_layout()
    plt.savefig(f"plots/{num_nodes}_2d_avg_norm_cost_diff.png")
    plt.show()

if __name__ == "__main__":
    results_folder = "C:/Projects/Heuristics for combinatorial optimisation/Heuristics-for-combinatorial-optimisation/results"
    num_nodes = 5000
    color_set_size = None
    # plot_3d_final_costs(results_folder, filter_num_nodes=num_nodes, filter_color_set_size=color_set_size)
    # plot_3d_normalized_costs(results_folder, filter_num_nodes=num_nodes, filter_color_set_size=color_set_size)
    # plot_avg_norm_cost_diff(results_folder, num_nodes=num_nodes)
    plot_3d_avg_norm_cost_diff(results_folder, num_nodes)
    # plot_2d_avg_norm_cost_diff(results_folder, num_nodes)
    