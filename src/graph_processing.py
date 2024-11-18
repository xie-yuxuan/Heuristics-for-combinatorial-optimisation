import json
import os
import numpy as np
import networkx as nx
import time
import copy
from networkx.readwrite import json_graph

from visualisation import draw_graph, plot_cost_data
from algorithms import optimise, optimise2, optimise3, optimise4
from graph_gen import generate_random_regular_graph

'''
Run this program to run both greedy and reluctant optimisation fn on graph for all initial colorings and save results to JSON.
fg and fr are transformation functions for cost change matrix for greedy and reluctant algos.
Uncomment last section for testing purposes of the optimisation fn.
'''


def load_graph_from_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # load graph_name and attibutes
    graph_name = data["graph_name"]
    color_set_size = data["color_set_size"]
    degree = data["degree"]
    num_nodes = data["num_nodes"]
    gaussian_mean = data["gaussian_mean"]
    gaussian_variance = data["gaussian_variance"]
    initial_node_colors = data["initial_node_colors"]
    
    graph_data = data["graph_data"]
    graph = json_graph.node_link_graph(graph_data)

    # uncomment to get adj matrix
    # adj_matrix = nx.adjacency_matrix(graph, weight='weight').toarray()
    
    return graph, graph_name, color_set_size, degree, num_nodes, gaussian_mean, gaussian_variance, initial_node_colors

# matrix transformation fn, depending on algo
fg = lambda x: x # greedy transforamtion to cost change matrix
def fr(x): # reluctant transformation to cost change matrix
    # check if x is a np array
    if isinstance(x, np.ndarray):
        vectorized_func = np.vectorize(lambda x: 0.0 if x == 0 else 1.0 / x) # vectorize to handle each ele individually
        return vectorized_func(x)
    else:
        return 0.0 if x == 0 else 1.0 / x

if __name__ == '__main__':

    file_path = r"C:\Projects\Heuristics for combinatorial optimisation\Heuristics-for-combinatorial-optimisation\data\graphs\(100, 10, 8).json"
    graph, graph_name, color_set_size, degree, num_nodes, gaussian_mean, gaussian_variance, initial_node_colors = load_graph_from_json(file_path)
    # uncomment to visualise graph plot bef optimisation\
    # draw_graph(graph, pos=nx.spring_layout(graph, seed=1), 
    #            graph_name=graph_name, 
    #            iterations_taken=0, 
    #            cost_data=None,
    #            color_set_size=color_set_size, 
    #            degree=degree, 
    #            num_nodes=num_nodes, 
    #            gaussian_mean=gaussian_mean, 
    #            gaussian_variance=gaussian_variance
    #            )

    results = {
        "graph_name": graph_name,
        "degree" : degree,
        "num_nodes" : num_nodes,
        "color_set_size" : color_set_size,
        "gaussian_mean" : gaussian_mean,
        "gaussian_variance" : gaussian_variance,
        "cost_data" : {}
    }

    # # get list of greedy and reluctant op given list of initial colorings and graph J --------------------------------------------
    
    # loop through list of initial colorings, assign colors and run optimisation
    for i, intial_coloring in enumerate(initial_node_colors):
        # assign one of the initial colorings
        for node, color in enumerate(intial_coloring): #TODO: slow, looping through all nodes to recolor
            graph.nodes[node]['color'] = color

        # graph_copy1 = copy.deepcopy(graph)
        graph_copy1 = copy.deepcopy(graph)
        graph_g, final_cost_g, iterations_taken_g, cost_data_g = optimise4(graph, color_set_size, algo_func=fg) 
        graph_r, final_cost_r, iterations_taken_r, cost_data_r = optimise4(graph_copy1, color_set_size, algo_func=fr) 

        results["cost_data"][f"initial_coloring_{i}"] = {
            "cost_data_g": cost_data_g,
            "cost_data_r": cost_data_r
        }

        print(f"{i} initial coloring optimisation complete")

    graphs_path = r"C:\Projects\Heuristics for combinatorial optimisation\Heuristics-for-combinatorial-optimisation\results"

    with open(os.path.join(graphs_path, f"{graph_name}_results.json"), 'w') as f:
        json.dump(results, f, indent = 2)

    print(f"Saved results to {graphs_path}/{graph_name}_results.json")  

    # # testing optimisation code on one graph and one coloring -----------------------------------------------------------------

    # for node, color in enumerate(initial_node_colors[1]): # assign idx 0 initial colors
    #         graph.nodes[node]['color'] = color

    # start_time = time.time()

    # # graph, final_cost, iterations_taken, cost_data = optimise(graph, color_set_size, algo = 'reluctant')
    # # graph, final_cost, iterations_taken, cost_data = optimise2(graph, color_set_size, algo = 'reluctant')  
    # # graph_copy = copy.deepcopy(graph)
    # # graph_g, final_cost_g, iterations_taken_g, cost_data_g = optimise3(graph, color_set_size, algo_func=fg) 
    # # graph_r, final_cost_r, iterations_taken_r, cost_data_r = optimise3(graph, color_set_size, algo_func=fr) 
    # # graph_g, final_cost_g, iterations_taken_g, cost_data_g = optimise4(graph, color_set_size, algo_func=fg) 
    # graph_r, final_cost_r, iterations_taken_r, cost_data_r = optimise4(graph, color_set_size, algo_func=fr) 
    
    # print("--- %s seconds ---" % (time.time() - start_time))

    # # print(f"Final cost g: {final_cost_g}")
    # print(f"Final cost r: {final_cost_r}")
    # # print(iterations_taken_g)
    # print(iterations_taken_r)

    # # uncomment to visualise graph plot aft optimisation
    # # draw_graph(graph_r, pos=nx.spring_layout(graph_r, seed=1), 
    # #            graph_name=graph_name, 
    # #            iterations_taken=iterations_taken_r, 
    # #            cost_data=cost_data_r,
    # #            color_set_size=color_set_size, 
    # #            degree=degree, 
    # #            num_nodes=num_nodes, 
    # #            gaussian_mean=gaussian_mean, 
    # #            gaussian_variance=gaussian_variance
    # #            )