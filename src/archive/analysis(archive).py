import networkx as nx
import json


from graph_processing import load_graph_from_json
from algorithms import naive_greedy, animate_naive_greedy, naive_reluctant, animate_naive_reluctant
from utils import calc_cost
from visualisation import animate


if __name__ == '__main__':
    # Graph and color map paths
    # graph_1_json_path = 'C:/Projects/Heuristics for combinatorial optimisation/Heuristics-for-combinatorial-optimisation/data/sample_graphs/graph1.json'
    color_map_path = 'C:/Projects/Heuristics for combinatorial optimisation/Heuristics-for-combinatorial-optimisation/data/color_map.json'
    with open(color_map_path, 'r') as f:
        color_map = json.load(f)['color_map']


    # Load graph and color map
    # graph_1, graph_1_name = load_json_graph(graph_1_json_path)

    # Random graph
    # random_graph_path = 'C:/Projects/Heuristics for combinatorial optimisation/Heuristics-for-combinatorial-optimisation/data/sample_graphs/random_graph.json'
    # random_graph, random_graph_name = load_json_graph(random_graph_path)

    # Random1 graph
    test_path = "C:\Projects\Heuristics for combinatorial optimisation\Heuristics-for-combinatorial-optimisation\data\graphs\(100, 10, 4).json"
    graph, graph_name, color_set_size, degree, num_nodes, gaussian_mean, gaussian_variance, initial_node_colors = load_graph_from_json(test_path)

    initial_coloring = initial_node_colors[0]
    for node, color in enumerate(initial_coloring): #TODO: slow, looping through all nodes to recolor
        graph.nodes[node]['color'] = color

    # Initialise parameters
    # Define layout for graph visualiation, set vertex positions
    # pos = nx.spring_layout(graph_1, seed=4) # force-directed algo
    # pos = nx.circular_layout(graph) # circle
    # pos = nx.random_layout(graph)
    # pos = nx.shell_layout(graph) # vertices arranged in concentric circles
    # pos = nx.kamada_kawai_layout(graph) # forced-directed algo but diff
    # pos = nx.spectral_layout(graph) # use eigenvectors of graph Laplacian matrix
    # pos = nx.draw_planar(graph) # planar graph

    max_iterations = 500
    color_set_size = 4

    # Apply optimisation algo ------------------------------------------------

    # Graph 1

    # pos = nx.spring_layout(graph_1, seed=4)
    
    # Naive greedy
    # graph_1_naive_greedy, final_cost, iterations_taken, cost_data = naive_greedy(graph_1, color_set_size, max_iterations)
    # draw_graph(graph_1_naive_greedy, pos, graph_1_name, iterations_taken, cost_data)

    # animate(graph_1, color_set_size, max_iterations, pos, graph_1_name, algo='naive greedy')

    # Naive reluctant
    # graph_1_naive_reluctant, final_cost, iterations_taken, cost_data = naive_reluctant(graph_1, color_set_size, max_iterations)
    # draw_graph(graph_1_naive_reluctant, pos, graph_1_name, iterations_taken, cost_data)
    
    # animate(graph_1, color_set_size, max_iterations, pos, graph_1_name, algo='naive reluctant')

    # Random graph

    # pos = nx.spring_layout(random_graph, seed=0)

    # random_graph_naive_greedy, final_cost, iterations_taken, cost_data = naive_greedy(random_graph, color_set_size, max_iterations)
    # draw_graph(random_graph_naive_greedy, pos, random_graph_name, iterations_taken, cost_data)

    # animate(random_graph, color_set_size, max_iterations, pos, random_graph_name, algo='naive greedy')

    # random_graph_naive_reluctant, final_cost, iterations_taken, cost_data = naive_reluctant(random_graph, color_set_size, max_iterations)
    # draw_graph(random_graph_naive_reluctant, pos, random_graph_name, iterations_taken, cost_data)

    # animate(random_graph, color_set_size, max_iterations, pos, random_graph_name, algo='naive reluctant')

    # Random1 graph

    pos = nx.spring_layout(graph, seed=0)

    # random1_naive_greedy, final_cost, iterations_taken, cost_data = naive_greedy(random1, color_set_size, max_iterations)
    # draw_graph(random1_naive_greedy, pos, random1_name, iterations_taken, cost_data)

    animate(graph, color_set_size, max_iterations, pos, graph_name, algo='naive greedy')

    # random1_naive_reluctant, final_cost, iterations_taken, cost_data = naive_reluctant(random1, color_set_size, max_iterations)
    # draw_graph(random1_naive_reluctant, pos, random1_name, iterations_taken, cost_data)

    # animate(random1, color_set_size, max_iterations, pos, random1_name, algo='naive reluctant')

