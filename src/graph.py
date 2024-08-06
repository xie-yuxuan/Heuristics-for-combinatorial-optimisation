import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


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

def draw_graph(graph, graph_name):
    '''
    Draw the graph using Graphviz and save to an image file.
    '''
    # Define layout, set vertex positions
    pos = nx.spring_layout(graph, seed=4) # force-directed algo
    # pos = nx.circular_layout(graph) # circle
    # pos = nx.random_layout(graph)
    # pos = nx.shell_layout(graph) # vertices arranged in concentric circles
    # pos = nx.kamada_kawai_layout(graph) # forced-directed algo but diff
    # pos = nx.spectral_layout(graph) # use eigenvectors of graph Laplacian matrix
    # pos = nx.draw_planar(graph) # planar graph

    vertex_colors = [color_map.get(str(graph.nodes[node].get('color', 0)), 'gray') for node in graph.nodes]

    edge_weights = nx.get_edge_attributes(graph, 'weight')

    # Draw the graph
    plt.figure(figsize=(6, 6))

    nx.draw_networkx(graph, pos, with_labels=True, node_color=vertex_colors, node_size=500, edge_color='black', font_color='white', font_size=10)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_weights, rotate=False)

    plt.title(graph_name)
    plt.savefig(graph_name)
    plt.show()

def load_color_map(color_map_file):
    '''
    Load the color map from a JSON file.
    '''
    with open(color_map_file, 'r') as f:
        data = json.load(f)
    return data['color_map']


if __name__ == '__main__':
    # # Analysis ------------------------
    # Graph and color map paths
    graph_1_json_path = 'C:/Users/Yuxuan Xie/Desktop/Heuristics for combinatorial optimisation/Heuristics-for-combinatorial-optimisation/data/sample_graphs/graph1.json'
    color_map_path = 'C:/Users/Yuxuan Xie/Desktop/Heuristics for combinatorial optimisation/Heuristics-for-combinatorial-optimisation/data/color_map.json'

    graph_1, graph_1_name = load_json_graph(graph_1_json_path)
    color_map = load_color_map(color_map_path)

    draw_graph(graph_1, graph_1_name)
