import networkx as nx
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
from matplotlib.animation import PillowWriter
import seaborn as sns

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
    plt.rcParams.update({
        "font.size": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "figure.titlesize": 16
    })
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    color_map_path = 'C:/Projects/Heuristics for combinatorial optimisation/Heuristics-for-combinatorial-optimisation/data/color_map.json'
    with open(color_map_path, 'r') as f:
        color_map = json.load(f)['color_map']

    vertex_colors = [color_map.get(str(graph.nodes[node].get('color', 0)), 'gray') for node in graph.nodes] #TODO

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
        fontsize=14, 
        bbox=dict(facecolor='white', alpha=0.5, edgecolor='none')
        )

    ax[0].set_title(graph_name)

    fig.tight_layout()
    plt.tight_layout()
    # plt.savefig(graph_name)
    plt.show()

def generate_heatmap_of_color_changes(original_colors, changes, num_nodes, num_groups, color_map):
    # Initialize the matrix with initial colors
    iterations = len(changes)
    heatmap = np.full((num_nodes, iterations), -1)

    # Track current colors of nodes from the initial state

    current_colors = original_colors.copy()

    for iteration, (node, color) in enumerate(changes):

        current_colors[node] = color
        heatmap[:, iteration] = current_colors

    color_palette = [color_map[str(i)] for i in range(len(color_map))]
    cmap = ListedColormap(color_palette[:num_groups])

    # Plot the heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap, cmap=cmap, cbar=False)
    plt.xlabel("Iteration")
    plt.ylabel("Node Index")
    plt.title("Node Recoloring Heatmap")
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
