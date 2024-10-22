import networkx as nx
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

from algorithms import naive_greedy, animate_naive_greedy, naive_reluctant, animate_naive_reluctant
from utils import calc_cost

def draw_graph(graph, pos, graph_name, iterations_taken, cost_data):
    '''
    Draw graph on a given axis
    '''
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    num_nodes = len(graph.nodes) # Calc number of nodes to scale size of font and node proportionately


    color_map_path = 'C:/Projects/Heuristics for combinatorial optimisation/Heuristics-for-combinatorial-optimisation/data/color_map.json'
    with open(color_map_path, 'r') as f:
        color_map = json.load(f)['color_map']

    vertex_colors = [color_map.get(str(graph.nodes[node].get('color', 0)), 'gray') for node in graph.nodes]

    edge_weights = nx.get_edge_attributes(graph, 'weight')

    nx.draw_networkx(graph, pos, with_labels=True, node_color=vertex_colors, node_size=5000/num_nodes, edge_color='black', font_color='white', font_size=100/num_nodes, ax=ax[0])
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_weights, rotate=False, font_size=100/num_nodes, ax=ax[0])

    ax[0].text(
        0.95, 0.05, f'Iterations: {iterations_taken}\nCost: {calc_cost(graph)}\nColors used: {len(set(nx.get_node_attributes(graph, "color").values()))}', 
        horizontalalignment='right',
        verticalalignment='bottom', 
        transform=ax[0].transAxes,
        fontsize=9, 
        bbox=dict(facecolor='white', alpha=0.5, edgecolor='none')
        )

    ax[0].set_title(graph_name)

    if cost_data:
        # Unpack cost data
        iterations, costs = cost_data
        
        # Plot the cost vs. iteration graph on ax[1]
        ax[1].plot(iterations, costs, marker='o', linestyle='-', color='b')
        ax[1].set_xlabel('Iterations')
        ax[1].set_ylabel('Cost')
        ax[1].set_title('Cost vs. Iteration')
        ax[1].grid(True)
    else:
        # Clear the second subplot if no cost graph is needed
        ax[1].axis('off')

    fig.tight_layout()
    # plt.tight_layout()
    # plt.savefig(graph_name)
    plt.show()

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
    ani.save('random_graph_naive_reluctant.gif', writer=writer)
    
    plt.show()
