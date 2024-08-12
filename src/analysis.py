import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from graph import load_color_map, load_json_graph
from algorithms import naive_greedy, animate_naive_greedy
from utils import calc_cost

def draw_graph(graph, pos, graph_name, iterations_taken):
    '''
    Draw graph on a given axis
    '''
    fig, ax = plt.subplots(figsize=(6, 6))

    vertex_colors = [color_map.get(str(graph.nodes[node].get('color', 0)), 'gray') for node in graph.nodes]

    edge_weights = nx.get_edge_attributes(graph, 'weight')

    nx.draw_networkx(graph, pos, with_labels=True, node_color=vertex_colors, node_size=500, edge_color='black', font_color='white', font_size=10, ax=ax)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_weights, rotate=False, ax=ax)

    ax.text(
        0.95, 0.1, f'Iterations: {iterations_taken}\nCost: {calc_cost(graph)}\nColors used: {len(set(nx.get_node_attributes(graph, "color").values()))}', 
        horizontalalignment='right',
        verticalalignment='center', 
        transform=plt.gca().transAxes,
        fontsize=9, 
        bbox=dict(facecolor='white', alpha=0.5, edgecolor='none')
        )

    ax.set_title(graph_name)

    plt.savefig(graph_name)
    plt.show()

def animate(graph, color_set_size, iterations, pos, graph_name, algo):
    """
    Animate graph coloring for a specific optimization algorithm.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    def update(frame_data):
        graph, cur_cost, iteration_count, recolored_node = frame_data

        ax.clear()

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

        nx.draw_networkx(graph, pos, with_labels=True, node_color=vertex_colors, node_size=500, edge_color=edge_colors, font_color='white', font_size=10, ax=ax)
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_weights, rotate=False, ax=ax)

        ax.text(
            0.95, 0.1, f'Iterations: {iteration_count}\nCost: {calc_cost(graph)}\nColors used: {len(set(nx.get_node_attributes(graph, "color").values()))}', 
            horizontalalignment='right',
            verticalalignment='center', 
            transform=ax.transAxes,
            fontsize=9, 
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='none')
        )

        ax.set_title(graph_name)

    if algo == 'naive greedy':
        # Create an animation
        ani = animation.FuncAnimation(
            fig, update, frames=animate_naive_greedy(graph, color_set_size, iterations), interval=500, repeat=False
        )
        plt.show()

if __name__ == '__main__':
    # Graph and color map paths
    graph_1_json_path = 'C:/Users/Yuxuan Xie/Desktop/Heuristics for combinatorial optimisation/Heuristics-for-combinatorial-optimisation/data/sample_graphs/graph1.json'
    color_map_path = 'C:/Users/Yuxuan Xie/Desktop/Heuristics for combinatorial optimisation/Heuristics-for-combinatorial-optimisation/data/color_map.json'

    # Load graph and color map
    graph_1, graph_1_name = load_json_graph(graph_1_json_path)
    color_map = load_color_map(color_map_path)

    # Initialise parameters
    # Define layout for graph visualiation, set vertex positions
    pos = nx.spring_layout(graph_1, seed=4) # force-directed algo
    # pos = nx.circular_layout(graph) # circle
    # pos = nx.random_layout(graph)
    # pos = nx.shell_layout(graph) # vertices arranged in concentric circles
    # pos = nx.kamada_kawai_layout(graph) # forced-directed algo but diff
    # pos = nx.spectral_layout(graph) # use eigenvectors of graph Laplacian matrix
    # pos = nx.draw_planar(graph) # planar graph

    max_iterations = 10
    color_set_size = 3

    # Apply optimisation algo
    graph_1_naive_greedy, cost, iterations_taken = naive_greedy(graph_1, color_set_size, max_iterations)
    draw_graph(graph_1_naive_greedy, pos, graph_1_name, iterations_taken)

    # Animate optimisation algo
    # animate(graph_1, color_set_size, max_iterations, pos, graph_1_name, algo='naive greedy')
