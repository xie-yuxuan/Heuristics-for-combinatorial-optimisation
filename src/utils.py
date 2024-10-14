import networkx as nx
import numpy as np
import json
import random

"""
Utility functions including:
- calc_cost
- calc_delta_cost
- generate_random_graph
"""

def calc_cost(graph):
    cost = 0
    
    vertex_colors = nx.get_node_attributes(graph, 'color')
    
    for vertex_1, vertex_2, edge_data in graph.edges(data=True):
        if vertex_colors.get(vertex_1) == vertex_colors.get(vertex_2): # Check if connected vertices have the same color
            cost += edge_data.get('weight')

    return cost

def calc_delta_cost(graph, vertex, ori_color, new_color):
    """
    Calc cost savings (delta) when a vertex is recolored
    """
    delta = 0

    for neighbor in graph.neighbors(vertex):
        neighbor_color = graph.nodes[neighbor]['color']

        if ori_color == neighbor_color:
            delta += graph[vertex][neighbor].get('weight')

        if new_color == neighbor_color:
            delta -= graph[vertex][neighbor].get('weight')

    return delta


def generate_random_graph(num_vertices, num_edges, parameter=None):
    """
    Generate and save a random graph in a JSON format, based on input parameters num_vertices and num_edges.

    Initial starting colors are randomised and weights are taken from a gaussian distribution. 
    """
    # Ensure that the number of edges is enough to create a connected graph
    if num_edges < num_vertices - 1:
        raise ValueError('Number of edges must be at least num_vertices - 1 to form a connected graph.')
    
    G = nx.Graph()
    
    G.add_nodes_from(range(num_vertices))
    
    # Generate a spanning tree to ensure the graph is connected
    spanning_tree_edges = list(nx.utils.pairwise(random.sample(range(num_vertices), num_vertices)))
    
    # Add the spanning tree edges to the graph
    for u, v in spanning_tree_edges:
        G.add_edge(u, v, weight=random.randint(1, 10))  # Random weight between 1 and 10
    
    # Calculate remaining edges to add
    remaining_edges_to_add = num_edges - (num_vertices - 1)
    
    # Generate additional unique edges
    all_possible_edges = [(i, j) for i in range(num_vertices) for j in range(i + 1, num_vertices)
                          if not G.has_edge(i, j)]
    random.shuffle(all_possible_edges)
    
    # Add the remaining edges
    for u, v in all_possible_edges[:remaining_edges_to_add]:
        G.add_edge(u, v, weight=random.randint(1, 10)) 
    
    nx.set_node_attributes(G, 0, 'color') # Set initial node colors to red (0)
    
    adjacency_matrix = nx.to_numpy_array(G).tolist()  # Creates a list of lists

    vertex_colors = nx.get_node_attributes(G, 'color') # Create vertex colors dictionary
    
    graph_data = {
        "name": "Random Graph",
        "adjacency_matrix": adjacency_matrix,
        "vertex_colors": vertex_colors
    }
    
    with open('random_graph.json', 'w') as f:
        json_string = json.dumps(graph_data, indent=2)
        json_string = json_string.replace('[\n        ', '[')
        json_string = json_string.replace('\n      ],', '],')
        
        f.write(json_string)
    
    return G
def generate_random_random_graph(num_vertices, num_edges, color_set_size, parameter=None):
    """
    Generate and save a random graph in a JSON format, based on input parameters num_vertices and num_edges.

    Initial starting colors are randomised and weights are taken from a gaussian distribution. 
    """
    # Ensure that the number of edges is enough to create a connected graph
    if num_edges < num_vertices - 1:
        raise ValueError('Number of edges must be at least num_vertices - 1 to form a connected graph.')
    
    G = nx.Graph()
    
    G.add_nodes_from(range(num_vertices))
    
    # Generate a spanning tree to ensure the graph is connected
    spanning_tree_edges = list(nx.utils.pairwise(random.sample(range(num_vertices), num_vertices)))
    
    # Add the spanning tree edges to the graph
    for u, v in spanning_tree_edges:
        #G.add_edge(u, v, weight=random.randint(1, 10))  # Random weight between 1 to 19
        G.add_edge(u, v, weight=random.randint(1, 10))  # Random weight drawn from a gaussian distribution
    
    # Calculate remaining edges to add
    remaining_edges_to_add = num_edges - (num_vertices - 1)
    
    # Generate additional unique edges
    all_possible_edges = [(i, j) for i in range(num_vertices) for j in range(i + 1, num_vertices)
                          if not G.has_edge(i, j)]
    random.shuffle(all_possible_edges)
    
    # Add the remaining edges
    for u, v in all_possible_edges[:remaining_edges_to_add]:
        G.add_edge(u, v, weight=random.randint(1, 10)) 
    
    # nx.set_node_attributes(G, 0, 'color') # Set initial node colors to red (0)
    node_colors = {i: random.randint(0, color_set_size - 1) for i in G.nodes()}
    nx.set_node_attributes(G, node_colors, 'color') # Randomise initial node colors 
    
    adjacency_matrix = nx.to_numpy_array(G).tolist()  # Creates a list of lists

    vertex_colors = nx.get_node_attributes(G, 'color') # Create vertex colors dictionary

    print(adjacency_matrix)
    print(vertex_colors)
    
    graph_data = {
        "name": "random1",
        "adjacency_matrix": adjacency_matrix,
        "vertex_colors": vertex_colors
    }
    
    with open('random1.json', 'w') as f:
        json.dump(graph_data, f, indent=2, separators=(',', ': '))
    
    return G

if __name__ == '__main__':

    # Example usage
    num_vertices = 100
    num_edges = 150
    graph = generate_random_random_graph(num_vertices, num_edges, 4)
    # generate_random_graph(num_vertices,num_edges)