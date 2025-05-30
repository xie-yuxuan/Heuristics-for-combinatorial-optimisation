from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from collections import defaultdict
from scipy.interpolate import griddata
import pyvista as pv
import seaborn as sns

from analysis_single_graph import avg_norm_cost_diff

# def plot_3d_final_costs(results_folder, filter_num_nodes, filter_color_set_size):
#     """
#     Plot a 3D graph of lowest final costs when no filter is specified, or a 2D graph when filtered by num_nodes or color_set_size.
#     """
#     import os
#     import json
#     import numpy as np
#     import matplotlib.pyplot as plt

#     # Degree-specific colors
#     degree_colors = {
#         20: 'red',
#         10: 'green',
#         5: 'blue',
#         2: 'orange'
#     }

#     # Collect data from files
#     data_points = []
#     result_files = [f for f in os.listdir(results_folder) if not f.endswith("uniform')_results.json")]

#     for result_file in result_files:
#         # Extract parameters (num_nodes, degree, color_set_size) from filename
#         file_name_parts = result_file.split('_')[0].strip('()').split(', ')
#         num_nodes = int(file_name_parts[0])
#         degree = int(file_name_parts[1])
#         color_set_size = int(file_name_parts[2])

#         # Check filters
#         if filter_num_nodes is not None and num_nodes != filter_num_nodes:
#             continue
#         if filter_color_set_size is not None and color_set_size != filter_color_set_size:
#             continue

#         # Load data from the file
#         file_path = os.path.join(results_folder, result_file)
#         with open(file_path, 'r') as f:
#             cost_data = json.load(f)["cost_data"]

#         final_costs_greedy = []
#         final_costs_reluctant = []

#         for initial_coloring_key, iteration_data in cost_data.items():
#             cost_data_g = iteration_data["cost_data_g"]
#             cost_data_r = iteration_data["cost_data_r"]

#             final_cost_g = cost_data_g[-1]
#             final_cost_r = cost_data_r[-1]

#             final_costs_greedy.append(final_cost_g)
#             final_costs_reluctant.append(final_cost_r)

#         # Store data points (lowest final costs for each approach)
#         min_final_cost_greedy = np.min(final_costs_greedy)
#         min_final_cost_reluctant = np.min(final_costs_reluctant)
#         data_points.append((num_nodes, degree, color_set_size, min_final_cost_greedy, min_final_cost_reluctant))

#     # If no data points match the filter, exit
#     if not data_points:
#         print("No data points match the specified filters.")
#         return

#     # Separate data for plotting
#     xs = np.array([d[0] for d in data_points])  # num_nodes
#     ys = np.array([d[2] for d in data_points])  # color_set_size
#     degrees = np.array([d[1] for d in data_points])  # degree
#     zs_greedy = np.array([d[3] for d in data_points])  # final costs (greedy)
#     zs_reluctant = np.array([d[4] for d in data_points])  # final costs (reluctant)

#     # Check if a 2D plot is needed
#     if filter_num_nodes is not None or filter_color_set_size is not None:
#         # Determine x-axis (color_set_size if num_nodes is fixed, otherwise num_nodes)
#         if filter_num_nodes is not None:
#             x_axis = ys
#             x_label = "Color Set Size"
#         else:
#             x_axis = xs
#             x_label = "Number of Nodes"

#         # Create 2D plot
#         plt.figure(figsize=(10, 6))
#         for degree, color in degree_colors.items():
#             # Filter points for the current degree
#             mask = (degrees == degree)
#             if not mask.any():
#                 continue

#             x_points = x_axis[mask]
#             y_greedy = zs_greedy[mask]
#             y_reluctant = zs_reluctant[mask]

#             # Sort the data by the number of nodes (x_points)
#             sorted_indices = np.argsort(x_points)
#             x_points_sorted = x_points[sorted_indices]
#             y_greedy_sorted = y_greedy[sorted_indices]
#             y_reluctant_sorted = y_reluctant[sorted_indices]

#             # Plot points
#             plt.plot(x_points_sorted, y_greedy_sorted, color=color, label=f"Greedy (Degree {degree})", marker='o', linestyle='-', markersize=8)
#             plt.plot(x_points_sorted, y_reluctant_sorted, color=color, label=f"Reluctant (Degree {degree})", marker='x', linestyle='--', markersize=8)

#         plt.xlabel(x_label)
#         plt.ylabel("Lowest Final Cost")
#         if filter_num_nodes is not None:
#             plt.title(f"Plot of Lowest Final Costs vs. {x_label} for num_node = {filter_num_nodes}")
#         else:
#             plt.title(f"Plot of Lowest Final Costs vs. {x_label} for color_set_size = {filter_color_set_size}")
#         plt.legend(loc='upper right', fontsize='small')
#         plt.grid(True)
#         plt.savefig(f"plots/({filter_num_nodes}, all, {filter_color_set_size})_lowest_cost.png")
#         plt.show()
#     else:
#         # Create 3D plot
#         fig = plt.figure(figsize=(12, 8))
#         ax = fig.add_subplot(111, projection='3d')

#         # Plot data for each degree
#         for degree, color in degree_colors.items():
#             # Filter points for the current degree
#             mask = (degrees == degree)
#             if not mask.any():
#                 continue  # Skip if no points for this degree

#             xs_degree = xs[mask]
#             ys_degree = ys[mask]
#             zs_greedy_degree = zs_greedy[mask]
#             zs_reluctant_degree = zs_reluctant[mask]

#             # Plot points for greedy and reluctant
#             ax.scatter(xs_degree, ys_degree, zs_greedy_degree, color=color, label=f"Greedy (Degree {degree})", marker='o')
#             ax.scatter(xs_degree, ys_degree, zs_reluctant_degree, color=color, label=f"Reluctant (Degree {degree})", marker='x')

#         # Set axis labels
#         ax.set_xlabel('Number of Nodes')
#         ax.set_ylabel('Color Set Size')
#         ax.set_zlabel('Lowest Final Cost')
#         ax.set_title('3D Plot of Lowest Final Costs by Degree')
#         ax.legend(loc='upper right', fontsize='small')
#         plt.savefig(f"plots/(all, all, all)_lowest_cost.png")
#         plt.show()


# def plot_3d_normalized_costs(results_folder, filter_num_nodes, filter_color_set_size):
#     """
#     Plot a 3D graph of the best normalized final costs when no filter is specified, 
#     or a 2D graph when filtered by num_nodes or color_set_size.
#     Normalized final cost is calculated as final_cost / num_nodes.
#     """

#     # Degree-specific colors
#     degree_colors = {
#         20: 'red',
#         10: 'green',
#         5: 'blue',
#         2: 'orange'
#     }

#     # Collect data from files
#     data_points = []
#     result_files = [f for f in os.listdir(results_folder) if not f.endswith("uniform')_results.json")]

#     for result_file in result_files:
#         # Extract parameters (num_nodes, degree, color_set_size) from filename
#         file_name_parts = result_file.split('_')[0].strip('()').split(', ')
#         num_nodes = int(file_name_parts[0])
#         degree = int(file_name_parts[1])
#         color_set_size = int(file_name_parts[2])

#         # Check filters
#         if filter_num_nodes is not None and num_nodes != filter_num_nodes:
#             continue
#         if filter_color_set_size is not None and color_set_size != filter_color_set_size:
#             continue

#         # Load data from the file
#         file_path = os.path.join(results_folder, result_file)
#         with open(file_path, 'r') as f:
#             cost_data = json.load(f)["cost_data"]

#         final_costs_greedy = []
#         final_costs_reluctant = []

#         for initial_coloring_key, iteration_data in cost_data.items():
#             cost_data_g = iteration_data["cost_data_g"]
#             cost_data_r = iteration_data["cost_data_r"]

#             final_cost_g = cost_data_g[-1]
#             final_cost_r = cost_data_r[-1]

#             final_costs_greedy.append(final_cost_g)
#             final_costs_reluctant.append(final_cost_r)

#         # Store data points (best normalized final costs for each approach)
#         best_final_cost_greedy = np.min(final_costs_greedy) / num_nodes
#         best_final_cost_reluctant = np.min(final_costs_reluctant) / num_nodes
#         data_points.append((num_nodes, degree, color_set_size, best_final_cost_greedy, best_final_cost_reluctant))

#     # If no data points match the filter, exit
#     if not data_points:
#         print("No data points match the specified filters.")
#         return

#     # Separate data for plotting
#     xs = np.array([d[0] for d in data_points])  # num_nodes
#     ys = np.array([d[2] for d in data_points])  # color_set_size
#     degrees = np.array([d[1] for d in data_points])  # degree
#     zs_greedy = np.array([d[3] for d in data_points])  # best normalized costs (greedy)
#     zs_reluctant = np.array([d[4] for d in data_points])  # best normalized costs (reluctant)

#     # Check if a 2D plot is needed
#     if filter_num_nodes is not None or filter_color_set_size is not None:
#         # Determine x-axis (color_set_size if num_nodes is fixed, otherwise num_nodes)
#         if filter_num_nodes is not None:
#             x_axis = ys
#             x_label = "Color Set Size"
#         else:
#             x_axis = xs
#             x_label = "Number of Nodes"

#         # Create 2D plot
#         plt.figure(figsize=(10, 6))
#         for degree, color in degree_colors.items():
#             # Filter points for the current degree
#             mask = (degrees == degree)
#             if not mask.any():
#                 continue

#             x_points = x_axis[mask]
#             y_greedy = zs_greedy[mask]
#             y_reluctant = zs_reluctant[mask]

#             # Sort the data by the number of nodes (x_points)
#             sorted_indices = np.argsort(x_points)
#             x_points_sorted = x_points[sorted_indices]
#             y_greedy_sorted = y_greedy[sorted_indices]
#             y_reluctant_sorted = y_reluctant[sorted_indices]

#             # Plot points
#             plt.plot(x_points_sorted, y_greedy_sorted, color=color, label=f"Greedy (Degree {degree})", marker='o', linestyle='-', markersize=8)
#             plt.plot(x_points_sorted, y_reluctant_sorted, color=color, label=f"Reluctant (Degree {degree})", marker='x', linestyle='--', markersize=8)

#         plt.xlabel(x_label)
#         plt.ylabel("Best Normalized Final Cost")
#         if filter_num_nodes is not None:
#             plt.title(f"Plot of Best Normalized Costs vs. {x_label} for num_node = {filter_num_nodes}")
#         else:
#             plt.title(f"Plot of Best Normalized Costs vs. {x_label} for color_set_size = {filter_color_set_size}")
#         plt.legend(loc='upper right', fontsize='small')
#         plt.grid(True)
#         plt.savefig(f"plots/({filter_num_nodes}, all, {filter_color_set_size})_best_norm_cost.png")
#         plt.show()
#     else:
#         # Create 3D plot
#         fig = plt.figure(figsize=(12, 8))
#         ax = fig.add_subplot(111, projection='3d')

#         # Plot data for each degree
#         for degree, color in degree_colors.items():
#             # Filter points for the current degree
#             mask = (degrees == degree)
#             if not mask.any():
#                 continue  # Skip if no points for this degree

#             xs_degree = xs[mask]
#             ys_degree = ys[mask]
#             zs_greedy_degree = zs_greedy[mask]
#             zs_reluctant_degree = zs_reluctant[mask]

#             # Plot points for greedy and reluctant
#             ax.scatter(xs_degree, ys_degree, zs_greedy_degree, color=color, label=f"Greedy (Degree {degree})", marker='o')
#             ax.scatter(xs_degree, ys_degree, zs_reluctant_degree, color=color, label=f"Reluctant (Degree {degree})", marker='x')

#         # Set axis labels
#         ax.set_xlabel('Number of Nodes')
#         ax.set_ylabel('Color Set Size')
#         ax.set_zlabel('Best Normalized Final Cost')
#         ax.set_title('3D Plot of Best Normalized Final Costs by Degree')
#         ax.legend(loc='upper right', fontsize='small')
#         plt.savefig(f"plots/(all, all, all)_best_norm_cost.png")
#         plt.show()





def plot_best_normalized_costs_for_uniform_edgeweight(results_folder, filter_num_nodes, filter_color_set_size):
    """
    Plot 2D graph when filtered by num_nodes or color_set_size.
    Normalized final cost is calculated as final_cost / num_nodes.
    """

    # Degree-specific colors
    degree_colors = {
        20: 'red',
        10: 'green',
        5: 'blue',
        2: 'orange'
    }

    # Collect data from files
    data_points = []
    result_files = [f for f in os.listdir(results_folder) if f.endswith("uniform')_results.json")]

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

            final_cost_g = cost_data_g[-1]
            final_cost_r = cost_data_r[-1]

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
    plt.savefig(f"plots/({filter_num_nodes}, all, {filter_color_set_size})_best_norm_cost_uniform_edgeweight.png")
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



# def plot_2d_avg_norm_cost_diff(results_folder, num_nodes):
#     '''
#     Plot 2D scatter plot of color set size vs degree with average normalized cost difference.
#     '''
#     # Get a list of all files in the results folder
#     result_files = [f for f in os.listdir(results_folder) if f.endswith('_results.json')]

#     # Data containers
#     data_points = []

#     # Iterate through all result files
#     for result_file in result_files:
#         # Extract graph parameters (num_nodes, degree, color_set_size) from the file name
#         file_name_parts = result_file.split('_')[0].strip('()').split(', ')
#         current_num_nodes = int(file_name_parts[0])
#         current_degree = int(file_name_parts[1])
#         current_color_set_size = int(file_name_parts[2])

#         # Only process the files for the specified num_nodes
#         if current_num_nodes == num_nodes:
#             # Construct the full file path
#             file_path = os.path.join(results_folder, result_file)
            
#             # Load the JSON data from the result file
#             with open(file_path, 'r') as f:
#                 cost_data = json.load(f)["cost_data"]
            
#             # Calculate the average normalized cost difference for this color set size and degree
#             avg_norm_cost_diff_value, _ = avg_norm_cost_diff(cost_data, num_nodes)
            
#             # Store the data for plotting
#             data_points.append([current_degree, current_color_set_size, avg_norm_cost_diff_value])

#     # Convert data points to numpy array for easier manipulation
#     data_points = np.array(data_points)
#     x_vals = data_points[:, 0]  # degree
#     y_vals = data_points[:, 1]  # color set size
#     z_vals = data_points[:, 2]  # avg normalized cost diff

#     # 2D Scatter Plot
#     plt.figure(figsize=(10, 8))
    
#     scatter = plt.scatter(x_vals, y_vals, c=z_vals, cmap='RdYlGn', marker='o', s=100)

#     plt.xlabel('Degree')
#     plt.ylabel('Color Set Size')
#     plt.title(f'2D Scatter: Avg Normalized Cost Diff (Num Nodes {num_nodes})')

#     # Add color scale
#     cbar = plt.colorbar(scatter)
#     cbar.set_label('Avg Normalized Cost Difference')
#     cbar.set_ticks(np.linspace(min(z_vals), max(z_vals), 5))  # Adjust color scale ticks

#     # Save the plot to a file
#     plt.tight_layout()
#     plt.savefig(f"plots/{num_nodes}_2d_avg_norm_cost_diff.png")
#     plt.show()

# --------------------------------------------------------------------------------------------

def plot_3d_plane_fitted_cost_diff(results_folder, num_nodes_fixed):
    """
    Plots a 3D scatter plot and a fitted surface (plane) of the normalised average
    final cost difference between greedy random and reluctant random, only for files
    with random_prob = 0, and for a fixed num_nodes.
    """

    data_points = []

    for filename in os.listdir(results_folder):
        if not filename.endswith("_results.json"):
            continue

        try:
            # Parse metadata from filename: (num_nodes, degree, colour_set_size, random_prob)
            name_parts = filename.strip("()_results.json").split(', ')
            num_nodes = int(name_parts[0])
            degree = int(name_parts[1])
            colour_set_size = int(name_parts[2])
            random_prob = float(name_parts[3])
        except:
            continue  # skip malformed filenames

        if num_nodes != num_nodes_fixed or random_prob != 0:
            continue

        # Load JSON
        file_path = os.path.join(results_folder, filename)
        with open(file_path, 'r') as f:
            data = json.load(f)

        cost_data = data.get("cost_data", {})
        gr_final_costs = []
        rr_final_costs = []

        for run in cost_data.values():
            if "cost_data_gr" in run and "cost_data_rr" in run:
                gr_final_costs.append(run["cost_data_gr"][-1])
                rr_final_costs.append(run["cost_data_rr"][-1])

        if gr_final_costs and rr_final_costs:
            avg_gr = np.mean(gr_final_costs)
            avg_rr = np.mean(rr_final_costs)
            norm_diff = (avg_gr - avg_rr) / num_nodes
            data_points.append((colour_set_size, degree, norm_diff))

    if not data_points:
        print("No valid data points found.")
        return

    # Unpack and interpolate
    xs, ys, zs = zip(*data_points)
    xs, ys, zs = np.array(xs), np.array(ys), np.array(zs)

    # Higher resolution mesh
    xi = np.linspace(min(xs), max(xs), 100)
    yi = np.linspace(min(ys), max(ys), 100)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((xs, ys), zs, (xi, yi), method='linear')

    # Plotting
    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Surface with reversed colormap (green high, red low)
    surf = ax.plot_surface(xi, yi, zi, alpha=0.6, cmap='RdYlGn', edgecolor='none')

    # Scatter points using same colormap
    scatter = ax.scatter(xs, ys, zs, c=zs, cmap='RdYlGn', s=60)

    # z = 0 plane
    x_plane, y_plane = np.meshgrid(np.unique(xs), np.unique(ys))
    z_plane = np.zeros_like(x_plane)
    ax.plot_surface(x_plane, y_plane, z_plane, alpha=0.2, color='grey')

    ax.set_xlabel("Colour Set Size")
    ax.set_ylabel("Degree")
    ax.set_zlabel("Normalised Avg Cost Difference")
    ax.set_title(f"Normalised Avg Final Cost Diff (Greedy - Reluctant), N = {num_nodes_fixed}")

    # Set camera angle: higher colour set size + lower degree corner in front
    ax.view_init(elev=30, azim=-60)

    # Add colour bar
    fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10, label='Normalised Cost Difference')

    # Save figure
    os.makedirs("plots", exist_ok=True)
    save_path = f"plots/({num_nodes_fixed}, all)_3D_plot.png"
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to: {save_path}")

    plt.tight_layout()
    plt.show()

def plot_2d_scatter_with_contours_and_labels(results_folder, num_nodes_fixed):
    """
    Plots a 2D scatter plot of normalised average cost difference between greedy random
    and reluctant random for each (degree, colour set size) point. Adds a decision boundary,
    smoothed background contours (green = positive, red = negative), and annotated values.
    """

    data_points = []

    for filename in os.listdir(results_folder):
        if not filename.endswith("_results.json"):
            continue

        try:
            name_parts = filename.strip("()_results.json").split(', ')
            num_nodes = int(name_parts[0])
            degree = int(name_parts[1])
            colour_set_size = int(name_parts[2])
            random_prob = float(name_parts[3])
        except:
            continue

        if num_nodes != num_nodes_fixed or random_prob != 0:
            continue

        file_path = os.path.join(results_folder, filename)
        with open(file_path, 'r') as f:
            data = json.load(f)

        cost_data = data.get("cost_data", {})
        gr_final_costs = []
        rr_final_costs = []

        for run in cost_data.values():
            if "cost_data_gr" in run and "cost_data_rr" in run:
                gr_final_costs.append(run["cost_data_gr"][-1])
                rr_final_costs.append(run["cost_data_rr"][-1])

        if gr_final_costs and rr_final_costs:
            avg_gr = np.mean(gr_final_costs)
            avg_rr = np.mean(rr_final_costs)
            norm_diff = (avg_gr - avg_rr) / num_nodes
            data_points.append((colour_set_size, degree, norm_diff))

    if not data_points:
        print("No valid data points found.")
        return

    # Unpack
    xs, ys, zs = zip(*data_points)
    xs, ys, zs = np.array(xs), np.array(ys), np.array(zs)

    # Grid for background
    xi = np.linspace(min(xs), max(xs), 200)
    yi = np.linspace(min(ys), max(ys), 200)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((xs, ys), zs, (xi, yi), method='cubic')

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    # Use flipped colormap: green = positive, red = negative
    cmap = 'RdYlGn'

    # Background colour contours
    contourf = ax.contourf(xi, yi, zi, levels=100, cmap=cmap, alpha=0.6)

    # Decision boundary where z = 0
    contour = ax.contour(xi, yi, zi, levels=[0], colors='black', linewidths=2)
    contour.collections[0].set_label("Decision Boundary (Cost Diff = 0)")

    # Scatter plot with colour
    scatter = ax.scatter(xs, ys, c=zs, cmap=cmap, s=100, edgecolors='black', zorder=10)

    # Annotate each point with value
    for x, y, z in zip(xs, ys, zs):
        ax.text(x, y, f"{z:.3f}", ha='center', va='center', fontsize=8, zorder=11)

    # Colour bar
    cbar = fig.colorbar(contourf, ax=ax)
    cbar.set_label("Normalised Cost Difference (Greedy - Reluctant)")

    ax.set_xlabel("Colour Set Size")
    ax.set_ylabel("Degree")
    ax.set_title(f"2D Scatter Plot with Contours and Decision Boundary, N = {num_nodes_fixed}")
    ax.legend()

    # Save plot
    os.makedirs("plots", exist_ok=True)
    save_path = f"plots/({num_nodes_fixed}, all)_2D_scatter_contour_labels.png"
    plt.savefig(save_path, dpi=300)
    print(f"2D annotated scatter plot with contours saved to: {save_path}")

    plt.tight_layout()
    plt.show()

def plot_3d_plane_fitted_cost_diff_SA(results_folder, num_nodes_fixed):
    """
    Plots a 3D scatter plot and a fitted surface (plane) of the normalised average
    final cost difference between greedy random and reluctant random, only for files
    with random_prob = 0, and for a fixed num_nodes.
    """

    data_points = []

    for filename in os.listdir(results_folder):
        if not filename.endswith("_results.json"):
            continue

        try:
            # Parse metadata from filename: (num_nodes, degree, colour_set_size, random_prob)
            name_parts = filename.strip("()_results.json").split(', ')
            num_nodes = int(name_parts[0])
            degree = int(name_parts[1])
            colour_set_size = int(name_parts[2])
            random_prob = float(name_parts[3])
        except:
            continue  # skip malformed filenames

        if num_nodes != num_nodes_fixed or random_prob != 0:
            continue

        # Load JSON
        file_path = os.path.join(results_folder, filename)
        with open(file_path, 'r') as f:
            data = json.load(f)

        cost_data = data.get("cost_data", {})
        gr_final_costs = []
        rr_final_costs = []

        for run in cost_data.values():
            if "cost_data_gsa" in run and "cost_data_rsa" in run:
                gr_final_costs.append(run["cost_data_gsa"][-1])
                rr_final_costs.append(run["cost_data_rsa"][-1])

        if gr_final_costs and rr_final_costs:
            avg_gr = np.mean(gr_final_costs)
            avg_rr = np.mean(rr_final_costs)
            norm_diff = (avg_gr - avg_rr) / num_nodes
            data_points.append((colour_set_size, degree, norm_diff))

    if not data_points:
        print("No valid data points found.")
        return

    # Unpack and interpolate
    xs, ys, zs = zip(*data_points)
    xs, ys, zs = np.array(xs), np.array(ys), np.array(zs)

    # Higher resolution mesh
    xi = np.linspace(min(xs), max(xs), 100)
    yi = np.linspace(min(ys), max(ys), 100)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((xs, ys), zs, (xi, yi), method='linear')

    # Plotting
    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Surface with reversed colormap (green high, red low)
    surf = ax.plot_surface(xi, yi, zi, alpha=0.6, cmap='RdYlGn', edgecolor='none')

    # Scatter points using same colormap
    scatter = ax.scatter(xs, ys, zs, c=zs, cmap='RdYlGn', s=60)

    # z = 0 plane
    x_plane, y_plane = np.meshgrid(np.unique(xs), np.unique(ys))
    z_plane = np.zeros_like(x_plane)
    ax.plot_surface(x_plane, y_plane, z_plane, alpha=0.2, color='grey')

    ax.set_xlabel("Colour Set Size")
    ax.set_ylabel("Degree")
    ax.set_zlabel("Normalised Avg Cost Difference")
    ax.set_title(f"Normalised Avg Final Cost Diff (Greedy - Reluctant), N = {num_nodes_fixed}")

    # Set camera angle: higher colour set size + lower degree corner in front
    ax.view_init(elev=30, azim=-60)

    # Add colour bar
    fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10, label='Normalised Cost Difference')

    # Save figure
    os.makedirs("plots", exist_ok=True)
    save_path = f"plots/({num_nodes_fixed}, all)_3D_plot_SA.png"
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to: {save_path}")

    plt.tight_layout()
    plt.show()

def plot_2d_scatter_with_contours_and_labels_SA(results_folder, num_nodes_fixed):
    """
    Plots a 2D scatter plot of normalised average cost difference between greedy random
    and reluctant random for each (degree, colour set size) point. Adds a decision boundary,
    smoothed background contours (green = positive, red = negative), and annotated values.
    """

    data_points = []

    for filename in os.listdir(results_folder):
        if not filename.endswith("_results.json"):
            continue

        try:
            name_parts = filename.strip("()_results.json").split(', ')
            num_nodes = int(name_parts[0])
            degree = int(name_parts[1])
            colour_set_size = int(name_parts[2])
            random_prob = float(name_parts[3])
        except:
            continue

        if num_nodes != num_nodes_fixed or random_prob != 0:
            continue

        file_path = os.path.join(results_folder, filename)
        with open(file_path, 'r') as f:
            data = json.load(f)

        cost_data = data.get("cost_data", {})
        gr_final_costs = []
        rr_final_costs = []

        for run in cost_data.values():
            if "cost_data_gsa" in run and "cost_data_rsa" in run:
                gr_final_costs.append(run["cost_data_gsa"][-1])
                rr_final_costs.append(run["cost_data_rsa"][-1])

        if gr_final_costs and rr_final_costs:
            avg_gr = np.mean(gr_final_costs)
            avg_rr = np.mean(rr_final_costs)
            norm_diff = (avg_gr - avg_rr) / num_nodes
            data_points.append((colour_set_size, degree, norm_diff))

    if not data_points:
        print("No valid data points found.")
        return

    # Unpack
    xs, ys, zs = zip(*data_points)
    xs, ys, zs = np.array(xs), np.array(ys), np.array(zs)

    # Grid for background
    xi = np.linspace(min(xs), max(xs), 200)
    yi = np.linspace(min(ys), max(ys), 200)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((xs, ys), zs, (xi, yi), method='cubic')

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    # Use flipped colormap: green = positive, red = negative
    cmap = 'RdYlGn'

    # Background colour contours
    contourf = ax.contourf(xi, yi, zi, levels=100, cmap=cmap, alpha=0.6)

    # Decision boundary where z = 0
    contour = ax.contour(xi, yi, zi, levels=[0], colors='black', linewidths=2)
    contour.collections[0].set_label("Decision Boundary (Cost Diff = 0)")

    # Scatter plot with colour
    scatter = ax.scatter(xs, ys, c=zs, cmap=cmap, s=100, edgecolors='black', zorder=10)

    # Annotate each point with value
    for x, y, z in zip(xs, ys, zs):
        ax.text(x, y, f"{z:.3f}", ha='center', va='center', fontsize=8, zorder=11)

    # Colour bar
    cbar = fig.colorbar(contourf, ax=ax)
    cbar.set_label("Normalised Cost Difference (Greedy - Reluctant)")

    ax.set_xlabel("Colour Set Size")
    ax.set_ylabel("Degree")
    ax.set_title(f"2D Scatter Plot with Contours and Decision Boundary, N = {num_nodes_fixed}")
    ax.legend()

    # Save plot
    os.makedirs("plots", exist_ok=True)
    save_path = f"plots/({num_nodes_fixed}, all)_2D_scatter_contour_labels_SA.png"
    plt.savefig(save_path, dpi=300)
    print(f"2D annotated scatter plot with contours saved to: {save_path}")

    plt.tight_layout()
    plt.show()


def plot_log_cost_vs_degree_randomised(results_folder, num_nodes_fixed, colour_set_size_fixed):
    """
    Plots log(final cost) vs degree for different random probabilities for GR and RR,
    including GSA and RSA. Uses distinct colours and markers for each configuration.
    """

    random_probs = [0, 0.1]
    degrees = [2, 6, 10, 14, 18]
    markers = ['o', 's', 'D', '^', 'v', 'P', '*']
    gr_colors = {0: 'red'}
    rr_colors = {0: 'green'}
    gsa_color = 'brown'
    rsa_color = 'blue'
    other_gr_color = 'orange'
    other_rr_color = 'purple'

    results = {
        'gr': {p: {} for p in random_probs},
        'rr': {p: {} for p in random_probs},
        'gsa': {},
        'rsa': {}
    }

    for filename in os.listdir(results_folder):
        if not filename.endswith("_results.json"):
            continue

        try:
            name_parts = filename.strip("()_results.json").split(', ')
            num_nodes = int(name_parts[0])
            degree = int(name_parts[1])
            colour_set_size = int(name_parts[2])
            random_prob = float(name_parts[3])
        except:
            continue

        if num_nodes != num_nodes_fixed or colour_set_size != colour_set_size_fixed or degree not in degrees:
            continue

        file_path = os.path.join(results_folder, filename)
        with open(file_path, 'r') as f:
            data = json.load(f)

        cost_data = data.get("cost_data", {})
        gr_final, rr_final, gsa_final, rsa_final = [], [], [], []

        for run in cost_data.values():
            if "cost_data_gr" in run:
                gr_final.append(run["cost_data_gr"][-1])
            if "cost_data_rr" in run:
                rr_final.append(run["cost_data_rr"][-1])
            if "cost_data_gsa" in run:
                gsa_final.append(run["cost_data_gsa"][-1])
            if "cost_data_rsa" in run:
                rsa_final.append(run["cost_data_rsa"][-1])

        if gr_final:
            results['gr'].setdefault(random_prob, {})[degree] = np.log(np.mean(gr_final))
        if rr_final:
            results['rr'].setdefault(random_prob, {})[degree] = np.log(np.mean(rr_final))
        if gsa_final and random_prob == 0:
            results['gsa'][degree] = np.log(np.mean(gsa_final))
        if rsa_final and random_prob == 0:
            results['rsa'][degree] = np.log(np.mean(rsa_final))

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, prob in enumerate(random_probs):
        x_gr = sorted(results['gr'][prob].keys())
        y_gr = [results['gr'][prob][d] for d in x_gr]
        x_rr = sorted(results['rr'][prob].keys())
        y_rr = [results['rr'][prob][d] for d in x_rr]

        if prob == 0:
            ax.plot(x_gr, y_gr, color=gr_colors[0], label="Greedy Random (p=0)", marker=markers[i])
            ax.plot(x_rr, y_rr, color=rr_colors[0], label="Reluctant Random (p=0)", marker=markers[i])
        else:
            if y_gr:
                ax.plot(x_gr, y_gr, color=other_gr_color, label=f"Greedy Random (p={prob})", marker=markers[i])
            if y_rr:
                ax.plot(x_rr, y_rr, color=other_rr_color, label=f"Reluctant Random (p={prob})", marker=markers[i])

    if results['gsa']:
        x_gsa = sorted(results['gsa'].keys())
        y_gsa = [results['gsa'][d] for d in x_gsa]
        ax.plot(x_gsa, y_gsa, color=gsa_color, label="Greedy SA", marker='X', linestyle='--')

    if results['rsa']:
        x_rsa = sorted(results['rsa'].keys())
        y_rsa = [results['rsa'][d] for d in x_rsa]
        ax.plot(x_rsa, y_rsa, color=rsa_color, label="Reluctant SA", marker='X', linestyle='--')

    ax.set_xlabel("Degree")
    ax.set_ylabel("log(Final Cost)")
    ax.set_title(f"log(Final Cost) vs Degree for Colour Set Size = {colour_set_size_fixed}")
    ax.legend()
    ax.grid(True)

    os.makedirs("plots", exist_ok=True)
    save_path = f"plots/log_final_cost_vs_degree_color{colour_set_size_fixed}.png"
    plt.savefig(save_path, dpi=300)
    print(f"Log-scale cost plot saved to: {save_path}")

    plt.tight_layout()
    plt.show()

def plot_cost_vs_degree_randomised(results_folder, num_nodes_fixed, colour_set_size_fixed):
    """
    Plots final cost vs degree for different random probabilities for GR and RR,
    including GSA and RSA. Uses distinct colours and markers for each configuration.
    """

    # Define styling
    random_probs = [0, 0.1]
    degrees = [2, 6, 10, 14, 18]
    markers = ['o', 's', 'D', '^', 'v', 'P', '*']
    gr_colors = {0: 'red'}
    rr_colors = {0: 'green'}
    gsa_color = 'brown'
    rsa_color = 'blue'
    other_gr_color = 'orange'
    other_rr_color = 'purple'

    # Collect results: structure {method: {prob: {degree: avg_final_cost}}}
    results = {
        'gr': {p: {} for p in random_probs},
        'rr': {p: {} for p in random_probs},
        'gsa': {},
        'rsa': {}
    }

    for filename in os.listdir(results_folder):
        if not filename.endswith("_results.json"):
            continue

        try:
            name_parts = filename.strip("()_results.json").split(', ')
            num_nodes = int(name_parts[0])
            degree = int(name_parts[1])
            colour_set_size = int(name_parts[2])
            random_prob = float(name_parts[3])
        except:
            continue

        if num_nodes != num_nodes_fixed or colour_set_size != colour_set_size_fixed or degree not in degrees:
            continue

        file_path = os.path.join(results_folder, filename)
        with open(file_path, 'r') as f:
            data = json.load(f)

        cost_data = data.get("cost_data", {})
        gr_final, rr_final, gsa_final, rsa_final = [], [], [], []

        for run in cost_data.values():
            if "cost_data_gr" in run:
                gr_final.append(run["cost_data_gr"][-1])
            if "cost_data_rr" in run:
                rr_final.append(run["cost_data_rr"][-1])
            if "cost_data_gsa" in run:
                gsa_final.append(run["cost_data_gsa"][-1])
            if "cost_data_rsa" in run:
                rsa_final.append(run["cost_data_rsa"][-1])

        if gr_final:
            results['gr'].setdefault(random_prob, {})[degree] = np.mean(gr_final)
        if rr_final:
            results['rr'].setdefault(random_prob, {})[degree] = np.mean(rr_final)
        if gsa_final and random_prob == 0:
            results['gsa'][degree] = np.mean(gsa_final)
        if rsa_final and random_prob == 0:
            results['rsa'][degree] = np.mean(rsa_final)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot GR and RR
    for i, prob in enumerate(random_probs):
        x_gr = sorted(results['gr'][prob].keys())
        y_gr = [results['gr'][prob][d] for d in x_gr]
        x_rr = sorted(results['rr'][prob].keys())
        y_rr = [results['rr'][prob][d] for d in x_rr]

        if prob == 0:
            ax.plot(x_gr, y_gr, color=gr_colors[0], label="Greedy Random (p=0)", marker=markers[i], alpha=0.5)
            ax.plot(x_rr, y_rr, color=rr_colors[0], label="Reluctant Random (p=0)", marker=markers[i], alpha=0.5)
        else:
            if y_gr:
                ax.plot(x_gr, y_gr, color=other_gr_color, label=f"Greedy Random (p={prob})", marker=markers[i], alpha=0.5)
            if y_rr:
                ax.plot(x_rr, y_rr, color=other_rr_color, label=f"Reluctant Random (p={prob})", marker=markers[i], alpha=0.5)

    # Plot GSA and RSA
    if results['gsa']:
        x_gsa = sorted(results['gsa'].keys())
        y_gsa = [results['gsa'][d] for d in x_gsa]
        ax.plot(x_gsa, y_gsa, color=gsa_color, label="Greedy SA", marker='X', linestyle='--', alpha=0.5)

    if results['rsa']:
        x_rsa = sorted(results['rsa'].keys())
        y_rsa = [results['rsa'][d] for d in x_rsa]
        ax.plot(x_rsa, y_rsa, color=rsa_color, label="Reluctant SA", marker='X', linestyle='--', alpha=0.5)

    ax.set_xlabel("Degree")
    ax.set_ylabel("Average Final Cost")
    ax.set_title(f"Final Cost vs Degree for Colour Set Size = {colour_set_size_fixed}")
    ax.legend()
    ax.grid(True)

    # Save
    os.makedirs("plots", exist_ok=True)
    save_path = f"plots/final_cost_vs_degree_color{colour_set_size_fixed}.png"
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to: {save_path}")

    plt.tight_layout()
    plt.show()


def plot_cost_vs_random_prob(results_folder, num_nodes_fixed, colour_set_size_fixed, degree_fixed):
    """
    Plots average final cost vs random probability for GR and RR at a fixed colour set size, degree, and num_nodes.
    Includes ±2 standard deviation shading.
    """

    random_probs = [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
    results = {
        'gr': {p: [] for p in random_probs},
        'rr': {p: [] for p in random_probs}
    }

    for filename in os.listdir(results_folder):
        if not filename.endswith("_results.json"):
            continue

        try:
            name_parts = filename.strip("()_results.json").split(', ')
            num_nodes = int(name_parts[0])
            degree = int(name_parts[1])
            colour_set_size = int(name_parts[2])
            random_prob = float(name_parts[3])
        except:
            continue

        if (num_nodes != num_nodes_fixed or 
            colour_set_size != colour_set_size_fixed or 
            degree != degree_fixed or 
            random_prob not in random_probs):
            continue

        file_path = os.path.join(results_folder, filename)
        with open(file_path, 'r') as f:
            data = json.load(f)

        cost_data = data.get("cost_data", {})
        gr_final, rr_final = [], []

        for run in cost_data.values():
            if "cost_data_gr" in run:
                gr_final.append(run["cost_data_gr"][-1])
            if "cost_data_rr" in run:
                rr_final.append(run["cost_data_rr"][-1])

        if gr_final:
            results['gr'][random_prob].extend(gr_final)
        if rr_final:
            results['rr'][random_prob].extend(rr_final)

    # Compute average and std dev final cost for each random prob
    gr_avg = [np.mean(results['gr'][p]) if results['gr'][p] else np.nan for p in random_probs]
    rr_avg = [np.mean(results['rr'][p]) if results['rr'][p] else np.nan for p in random_probs]
    gr_std = [np.std(results['gr'][p], ddof=1) if results['gr'][p] else np.nan for p in random_probs]
    rr_std = [np.std(results['rr'][p], ddof=1) if results['rr'][p] else np.nan for p in random_probs]

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(random_probs, gr_avg, color='orange', marker='o', label='Greedy Random')
    ax.plot(random_probs, rr_avg, color='purple', marker='s', label='Reluctant Random')
    ax.fill_between(random_probs, np.array(gr_avg) - 2 * np.array(gr_std), np.array(gr_avg) + 2 * np.array(gr_std),
                    color='orange', alpha=0.2, label='GR ±2 SD')
    ax.fill_between(random_probs, np.array(rr_avg) - 2 * np.array(rr_std), np.array(rr_avg) + 2 * np.array(rr_std),
                    color='purple', alpha=0.2, label='RR ±2 SD')

    ax.set_xlabel("Random Probability")
    ax.set_ylabel("Average Final Cost")
    ax.set_title(f"Average Final Cost vs Random Probability\nColour Set Size = {colour_set_size_fixed}, Degree = {degree_fixed}, Num Nodes = {num_nodes_fixed}")
    ax.legend()
    ax.grid(True)

    os.makedirs("plots", exist_ok=True)
    save_path = f"plots/avg_cost_vs_random_prob_color{colour_set_size_fixed}_deg{degree_fixed}.png"
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to: {save_path}")

    plt.tight_layout()
    plt.show()

def plot_iterations_vs_random_prob(results_folder, num_nodes_fixed=5000, colour_set_size_fixed=2, degree_fixed=10):
    """
    Plots the average number of iterations to convergence vs random probability
    for GR and RR on a fixed graph configuration: num_nodes=5000, degree=10, colour_set_size=2.
    Includes ±2 standard deviation shading.
    """

    random_probs = [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
    results = {
        'gr': {p: [] for p in random_probs},
        'rr': {p: [] for p in random_probs}
    }

    for filename in os.listdir(results_folder):
        if not filename.endswith("_results.json"):
            continue

        try:
            name_parts = filename.strip("()_results.json").split(', ')
            num_nodes = int(name_parts[0])
            degree = int(name_parts[1])
            colour_set_size = int(name_parts[2])
            random_prob = float(name_parts[3])
        except:
            continue

        if (num_nodes != num_nodes_fixed or 
            colour_set_size != colour_set_size_fixed or 
            degree != degree_fixed or 
            random_prob not in random_probs):
            continue

        file_path = os.path.join(results_folder, filename)
        with open(file_path, 'r') as f:
            data = json.load(f)

        cost_data = data.get("cost_data", {})

        for run in cost_data.values():
            if "cost_data_gr" in run:
                results['gr'][random_prob].append(len(run["cost_data_gr"]))
            if "cost_data_rr" in run:
                results['rr'][random_prob].append(len(run["cost_data_rr"]))

    # Compute average iterations and std dev
    gr_avg = [np.mean(results['gr'][p]) if results['gr'][p] else np.nan for p in random_probs]
    rr_avg = [np.mean(results['rr'][p]) if results['rr'][p] else np.nan for p in random_probs]
    gr_std = [np.std(results['gr'][p], ddof=1) if results['gr'][p] else np.nan for p in random_probs]
    rr_std = [np.std(results['rr'][p], ddof=1) if results['rr'][p] else np.nan for p in random_probs]

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(random_probs, gr_avg, color='orange', marker='o', label='Greedy Random')
    ax.plot(random_probs, rr_avg, color='purple', marker='s', label='Reluctant Random')
    ax.fill_between(random_probs, np.array(gr_avg) - 2 * np.array(gr_std), np.array(gr_avg) + 2 * np.array(gr_std),
                    color='orange', alpha=0.2, label='GR ±2 SD')
    ax.fill_between(random_probs, np.array(rr_avg) - 2 * np.array(rr_std), np.array(rr_avg) + 2 * np.array(rr_std),
                    color='purple', alpha=0.2, label='RR ±2 SD')

    ax.set_xlabel("Random Probability")
    ax.set_ylabel("Average Number of Iterations to Convergence")
    ax.set_title(f"Iterations vs Random Probability\nN={num_nodes_fixed}, Degree={degree_fixed}, Colour Set Size={colour_set_size_fixed}")
    ax.legend()
    ax.grid(True)

    os.makedirs("plots", exist_ok=True)
    save_path = f"plots/avg_iterations_vs_random_prob_N{num_nodes_fixed}_C{colour_set_size_fixed}_D{degree_fixed}.png"
    # plt.savefig(save_path, dpi=300)
    print(f"Plot saved to: {save_path}")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    results_folder = "C:/Projects/Heuristics for combinatorial optimisation/results"
    # plot_3d_plane_fitted_cost_diff(results_folder, 5000)
    # plot_2d_scatter_with_contours_and_labels(results_folder, 5000)
    # plot_3d_plane_fitted_cost_diff_SA(results_folder, 5000)
    # plot_2d_scatter_with_contours_and_labels_SA(results_folder, 5000)
    # plot_cost_vs_degree_randomised(results_folder, 5000, colour_set_size_fixed=2)
    plot_cost_vs_random_prob(results_folder, num_nodes_fixed=5000, colour_set_size_fixed=2, degree_fixed=10)
    # plot_iterations_vs_random_prob(results_folder)


    