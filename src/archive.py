# def generate_random_graph(num_vertices, num_edges, color_set_size, parameter=None):
#     """
#     Generate and save a random graph in a JSON format, based on input parameters num_vertices and num_edges.

#     Initial starting colors are randomised and weights are taken from a gaussian distribution. 
#     """
#     # Ensure that the number of edges is enough to create a connected graph
#     if num_edges < num_vertices - 1:
#         raise ValueError('Number of edges must be at least num_vertices - 1 to form a connected graph.')
    
#     G = nx.Graph()
    
#     G.add_nodes_from(range(num_vertices))
    
#     # Generate a spanning tree to ensure the graph is connected
#     spanning_tree_edges = list(nx.utils.pairwise(random.sample(range(num_vertices), num_vertices)))
    
#     # Add the spanning tree edges to the graph
#     for u, v in spanning_tree_edges:
#         #G.add_edge(u, v, weight=random.randint(1, 10))  # Random weight between 1 to 19
#         G.add_edge(u, v, weight=random.randint(1, 10))  # Random weight drawn from a gaussian distribution
    
#     # Calculate remaining edges to add
#     remaining_edges_to_add = num_edges - (num_vertices - 1)
    
#     # Generate additional unique edges
#     all_possible_edges = [(i, j) for i in range(num_vertices) for j in range(i + 1, num_vertices)
#                           if not G.has_edge(i, j)]
#     random.shuffle(all_possible_edges)
    
#     # Add the remaining edges
#     for u, v in all_possible_edges[:remaining_edges_to_add]:
#         G.add_edge(u, v, weight=random.randint(1, 10)) 
    
#     # nx.set_node_attributes(G, 0, 'color') # Set initial node colors to red (0)
#     node_colors = {i: random.randint(0, color_set_size - 1) for i in G.nodes()}
#     nx.set_node_attributes(G, node_colors, 'color') # Randomise initial node colors 
    
#     adjacency_matrix = nx.to_numpy_array(G).tolist()  # Creates a list of lists

#     vertex_colors = nx.get_node_attributes(G, 'color') # Create vertex colors dictionary

#     print(adjacency_matrix)
#     print(vertex_colors)
    
#     graph_data = {
#         "name": "random1",
#         "adjacency_matrix": adjacency_matrix,
#         "vertex_colors": vertex_colors
#     }
    
#     with open('random1.json', 'w') as f:
#         json.dump(graph_data, f, indent=2, separators=(',', ': '))
    
#     return G