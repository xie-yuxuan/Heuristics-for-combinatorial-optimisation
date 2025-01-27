import networkx as nx
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

from algorithms import naive_greedy, animate_naive_greedy, naive_reluctant, animate_naive_reluctant
from utils import calc_cost

'''
Visualisation functions for a single graph.
Multi graph visualisation functions are in the corresponding programs e.g. analysis_num_nodes.py.
'''

def draw_graph(graph, pos, graph_name, iterations_taken, cost_data, 
               color_set_size, 
               degree, 
               num_nodes, 
               gaussian_mean, 
               gaussian_variance, 
               ground_truth_log_likelihood):
    '''
    Draw graph on a given axis
    '''
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    color_map_path = 'C:/Projects/Heuristics for combinatorial optimisation/Heuristics-for-combinatorial-optimisation/data/color_map.json'
    with open(color_map_path, 'r') as f:
        color_map = json.load(f)['color_map']

    vertex_colors = [color_map.get(str(graph.nodes[node].get('color', 0)), 'gray') for node in graph.nodes]

    edge_weights = nx.get_edge_attributes(graph, 'weight')

    nx.draw_networkx(graph, pos, with_labels=True, node_color=vertex_colors, node_size=5000/num_nodes, edge_color='black', font_color='white', font_size=100/num_nodes, ax=ax[0])
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_weights, rotate=False, font_size=100/num_nodes, ax=ax[0])

    if cost_data:
        # Unpack cost data
        iterations, costs = cost_data
        
        # Plot the cost vs. iteration graph on ax[1]
        ax[1].plot(iterations, costs, marker='o', linestyle='-', color='b')
        ax[1].set_xlabel('Iterations')
        ax[1].set_ylabel('Cost')
        ax[1].set_title('Cost vs. Iteration')
        ax[1].grid(True)
        if ground_truth_log_likelihood:
            ax[1].axhline(y=ground_truth_log_likelihood, color='r', linestyle='--', label='Ground Truth')
            ax[1].text(0.5, ground_truth_log_likelihood, f'{ground_truth_log_likelihood:.2f}', 
               color='r', ha='center', va='bottom', fontsize=10, transform=ax[1].get_yaxis_transform())
    
    
    else:
        # Clear the second subplot if no cost graph is needed
        ax[1].axis('off')

    ax[0].text(
        0.98, 0.0, 
        f'Iterations: {iterations_taken}\n'
        # f'Cost: {costs[-1]}\n'
        f'Colors used: {len(set(nx.get_node_attributes(graph, "color").values()))}\n'
        f'Degree: {degree}\n'
        f'Number of nodes: {num_nodes}\n'
        f'Color set size: {color_set_size}\n'
        f'Gaussian Mean: {gaussian_mean}\n'
        f'Gaussian Variance: {gaussian_variance}\n',
        horizontalalignment='right',
        verticalalignment='bottom', 
        transform=ax[0].transAxes,
        fontsize=9, 
        bbox=dict(facecolor='white', alpha=0.5, edgecolor='none')
        )

    ax[0].set_title(graph_name)

    fig.tight_layout()
    plt.tight_layout()
    # plt.savefig(graph_name)
    plt.show()

def sbm_plot_cost_data(cost_data, graph_name, num_groups, num_nodes, group_mode, ground_truth_log_likelihood, specific_coloring):

    plt.figure(figsize=(10, 6))
    # Iterate over each initial coloring in cost_data
    for i, (key, value) in enumerate(cost_data.items()):
        # Skip other colorings if a specific coloring is selected
        if specific_coloring is not None and i != specific_coloring:
            continue

        # Extract data for the greedy and reluctant algorithms
        iterations_fg, costs_fg = value["cost_data_g"]
        iterations_fr, costs_fr = value["cost_data_r"]

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

    plt.axhline(y=ground_truth_log_likelihood, color='b', linestyle='--', label='Ground Truth')
    plt.text(0.5, ground_truth_log_likelihood, f'{ground_truth_log_likelihood:.2f}', 
               color='b', ha='center', va='bottom', fontsize=10)

    plt.plot([], [], color="red", label="Greedy")
    plt.plot([], [], color="green", label="Reluctant")
    plt.legend(loc="lower right")
    
    param_text = (f"Number of Groups: {num_groups}\n"
                  f"Number of Nodes: {num_nodes}\n"
                  f"Group Mode: {group_mode}")
    plt.gcf().text(0.66, 0.3, param_text, fontsize=8, bbox=dict(facecolor='white', alpha=0.5))

    plt.xlabel("Iterations")
    plt.ylabel("Log Likelihood")
    plt.title(f"Log Likelihood vs Iterations for Greedy and Reluctant on {graph_name}")
    plt.grid()

    plt.savefig(f"Heuristics-for-combinatorial-optimisation/plots/{graph_name}_cost.png")

    plt.show()


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
            plt.hlines(final_cost_fg, total_iterations_fg, total_iterations_fr, colors="blue", linestyles="dashed", alpha=0.3)

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

    # plt.savefig(f"plots/{graph_name}_cost.png")

    plt.show()


def sbm_plot_final_costs(cost_data, graph_name, num_nodes, num_groups, group_mode, ground_truth_log_likelihood):
    '''
    Plot a scatter plot of final log likelihood against initial coloring. 
    Show average final log likelihood.
    Show the ground truth log likelihood
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
    plt.axhline(ground_truth_log_likelihood, color='b', linestyle='--', label=f'Ground Truth: {ground_truth_log_likelihood}')
    plt.text(0.5, ground_truth_log_likelihood, f'{ground_truth_log_likelihood:.2f}', 
               color='b', ha='center', va='bottom', fontsize=10)

    plt.xlabel('Initial Coloring Index')
    plt.ylabel('Final Log Likelihood')
    plt.title(f'Greedy and Reluctant Final Log likelihood for All Initial Colorings of {graph_name}')

    experiment_text = (f"Number of Groups: {num_groups}\n"
                  f"Number of Nodes: {num_nodes}\n"
                  f"Group Mode: {group_mode}")
    
    plt.gca().text(0.05, 0.3, experiment_text, transform=plt.gca().transAxes, fontsize=8, 
                   verticalalignment='bottom', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.5))

    plt.legend(loc='lower left')

    plt.savefig(f"Heuristics-for-combinatorial-optimisation/plots/{graph_name}_cost.png")

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

# ----------------------------------- ANIMATION -----------------------------------------------

def animate(graph, color_set_size, iterations, pos, graph_name, algo):
    """
    Animate graph coloring for a specific optimization algorithm.
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Initialize lists to store cost data
    iteration_list = []
    cost_list = []

    def update(frame_data):
        graph, cur_cost, iteration_count, recolored_node = frame_data

        iteration_list.append(iteration_count)
        cost_list.append(cur_cost)

        ax[0].clear()
        ax[1].clear()

        color_map_path = 'C:/Projects/Heuristics for combinatorial optimisation/Heuristics-for-combinatorial-optimisation/data/color_map.json'
        with open(color_map_path, 'r') as f:
            color_map = json.load(f)['color_map']

        vertex_colors = [color_map.get(str(graph.nodes[node].get('color', 0)), 'gray') for node in graph.nodes]
        edge_colors = ['black'] * len(graph.edges)

        if recolored_node is not None: # Highlight recolored_node
            edge_colors = ['lightgray'] * len(graph.edges)
            
            recolored_node_color = color_map.get(str(graph.nodes[recolored_node].get('color', 0)), 'red')
            vertex_colors[recolored_node] = recolored_node_color

            # Update the edge colors for edges connected to the recolored node
            for neighbor in graph.neighbors(recolored_node):
                edge_key = (min(recolored_node, neighbor), max(recolored_node, neighbor))
                if edge_key in graph.edges:
                    edge_idx = list(graph.edges).index(edge_key)
                    edge_colors[edge_idx] = 'black'
        
        edge_weights = nx.get_edge_attributes(graph, 'weight')

        num_nodes = len(graph.nodes) # Calc number of nodes to scale size of font and node proportionately

        nx.draw_networkx(graph, pos, with_labels=True, node_color=vertex_colors, node_size=5000/num_nodes, edge_color=edge_colors, font_color='white', font_size=100/num_nodes, ax=ax[0])
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_weights, rotate=False, font_size=100/num_nodes, ax=ax[0])

        ax[0].text(
            0.95, 0.05, f'Iterations: {iteration_count}\nCost: {calc_cost(graph)}\nColors used: {len(set(nx.get_node_attributes(graph, "color").values()))}', 
            horizontalalignment='right',
            verticalalignment='center', 
            transform=ax[0].transAxes,
            fontsize=9, 
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='none')
        )

        ax[0].set_title(graph_name)

        if iteration_list:
            ax[1].plot(iteration_list, cost_list, marker='o', linestyle='-', color='b')
            ax[1].set_xlabel('Iterations')
            ax[1].set_ylabel('Cost')
            ax[1].set_title('Cost vs. Iteration')
            ax[1].grid(True)
        else:
            ax[1].axis('off')

    if algo == 'naive greedy':
        # Create an animation
        ani = animation.FuncAnimation(
            fig, update, frames=animate_naive_greedy(graph, color_set_size, iterations), interval=5, repeat=False
        )
    elif algo == 'naive reluctant':
        # Create an animation
        ani = animation.FuncAnimation(
            fig, update, frames=animate_naive_reluctant(graph, color_set_size, iterations), interval=5, repeat=False
        )

    fig.tight_layout()
    
    # Save animation as a gif
    writer = animation.PillowWriter(fps=10,
                                metadata=dict(artist='Me'),
                                bitrate=1800)
    # ani.save('test1_naive_reluctant.gif', writer=writer)
    
    plt.show()
