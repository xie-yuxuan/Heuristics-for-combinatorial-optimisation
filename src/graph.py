import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def load_json_graph(json_file):
    '''
    construct graph from adj matrix in the json file. Update vertex colors
    '''
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    adj_matrix = np.array(data['adjacency_matrix'])
    graph = nx.from_numpy_array(adj_matrix)

    vertex_colors = data.get('vertex_colors')
    vertex_colors = {int(k):v for k, v in vertex_colors.items()}
    nx.set_node_attributes(graph, vertex_colors, 'color')

    graph_name = data.get('name')

    return graph, graph_name

def load_color_map(color_map_file):
    '''
    Load the color map from a JSON file.
    '''
    with open(color_map_file, 'r') as f:
        data = json.load(f)
    return data['color_map']

def calc_cost(graph):
    cost = 0
    
    vertex_colors = nx.get_node_attributes(graph, 'color')
    
    for vertex_1, vertex_2, edge_data in graph.edges(data=True):
        if vertex_colors.get(vertex_1) == vertex_colors.get(vertex_2):  # Check if connected vertices have the same color
            cost += edge_data.get('weight')

    return cost

def calc_delta_cost(graph, vertex, ori_color, new_color):
    """
    Calc change in cost (delta) when a vertex is recolored
    """
    delta = 0

    for neighbor in graph.neighbors(vertex):
        neighbor_color = graph.nodes[neighbor]['color']

        if ori_color == neighbor_color:
            delta += graph[vertex][neighbor].get('weight')

        if new_color == neighbor_color:
            delta -= graph[vertex][neighbor].get('weight')

    return delta

def greedy(graph, color_set_size, iterations):
    # Initialise cost and color set
    cur_cost = calc_cost(graph)
    color_set = list(range(color_set_size))
    itertions_taken = 0
    
    # Greedy algo for loop based on fixed no. of iterations
    for i in range(iterations):
        # Initialise choice combination, determined by largest cost reduction
        vertex_choice = None
        color_choice = None
        max_cost_reduction = 0

        for vertex in graph.nodes: # graph.nodes is a list of all nodes
            # print(f'vertex {vertex}')
            ori_color = graph.nodes[vertex]['color']

            for color in color_set:
                # print(f'color {color}')
                delta_cost = calc_delta_cost(graph, vertex, ori_color, color)
                # print(f'delta_cost {delta_cost}')

                if delta_cost > max_cost_reduction:
                    max_cost_reduction = delta_cost
                    vertex_choice = vertex
                    color_choice = color
                    
        # print(f'max_cost_reduc {max_cost_reduction}')
        # print(f'vertex_choice {vertex_choice}')
        # print(f'color_choice {color_choice}')

        if max_cost_reduction == 0:
            break
        
        # Recoloring 
        graph.nodes[vertex_choice]['color'] = color_choice
        cur_cost -= max_cost_reduction
        iterations_taken = i + 1

        # Plot graph to show recoloring 
        draw_graph(graph, pos, graph_1_name, iterations_taken)

    return graph, cur_cost, itertions_taken


def animate_greedy(graph, color_set_size, iterations):
    cur_cost = calc_cost(graph)
    color_set = list(range(color_set_size))
    iterations_taken = 0

    yield graph, cur_cost, iterations_taken # yield initial state

    for i in range(iterations):
        vertex_choice = None
        color_choice = None
        max_cost_reduction = 0

        for vertex in graph.nodes:
            ori_color = graph.nodes[vertex]['color']

            for color in color_set:
                delta_cost = calc_delta_cost(graph, vertex, ori_color, color)

                if delta_cost > max_cost_reduction:
                    max_cost_reduction = delta_cost
                    vertex_choice = vertex
                    color_choice = color

        if max_cost_reduction == 0:
            break
        
        # Recoloring 
        graph.nodes[vertex_choice]['color'] = color_choice
        cur_cost -= max_cost_reduction
        iterations_taken = i + 1

        yield graph, cur_cost, iterations_taken # yield updated state


def reluctant(graph, color_set_size, iterations):
    pass

def draw_graph(graph, pos, graph_name, iteration_count):
    '''
    Draw the graph using Graphviz on the given axis
    '''

    vertex_colors = [color_map.get(str(graph.nodes[node].get('color', 0)), 'gray') for node in graph.nodes]

    edge_weights = nx.get_edge_attributes(graph, 'weight')

    nx.draw_networkx(graph, pos, with_labels=True, node_color=vertex_colors, node_size=500, edge_color='black', font_color='white', font_size=10)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_weights, rotate=False)

    plt.text(
        0.95, 0.1, f'Iterations: {iteration_count}\nCost: {calc_cost(graph)}\nColors used: {len(set(nx.get_node_attributes(graph, "color").values()))}', 
        horizontalalignment='right',
        verticalalignment='center', 
        transform=plt.gca().transAxes,
        fontsize=9, 
        bbox=dict(facecolor='white', alpha=0.5, edgecolor='none')
        )

    plt.title(graph_name)
    plt.savefig(graph_name)
    plt.show()


# Animtation functions

def animate(graph, color_set_size, iterations, pos, graph_name):
    fig, ax = plt.subplots(figsize=(6, 6))

    def update(frame_data):
        graph, cur_cost, iteration_count = frame_data
        ax.clear()

        vertex_colors = [color_map.get(str(graph.nodes[node].get('color', 0)), 'gray') for node in graph.nodes]
        edge_weights = nx.get_edge_attributes(graph, 'weight')

        nx.draw_networkx(graph, pos, with_labels=True, node_color=vertex_colors, node_size=500, edge_color='black', font_color='white', font_size=10, ax=ax)
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

    # Create an animation using the greedy function as a generator
    ani = animation.FuncAnimation(
        fig, update, frames=animate_greedy(graph, color_set_size, iterations), interval=1000, repeat=False
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

    # Apply optimising algorithm
    graph_1_greedy, cost, iterations_taken = greedy(graph_1, color_set_size, max_iterations)

    # Animation
    # animate(graph_1, color_set_size, max_iterations, pos, graph_1_name)

