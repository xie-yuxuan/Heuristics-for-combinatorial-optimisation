import json
import numpy as np
import networkx as nx

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
    Load the color map from a JSON file
    '''
    with open(color_map_file, 'r') as f:
        data = json.load(f)
    return data['color_map']








