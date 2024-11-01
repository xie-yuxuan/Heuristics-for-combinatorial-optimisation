import json
import numpy as np
import networkx as nx
import time
from networkx.readwrite import json_graph


from visualisation import draw_graph
from algorithms import optimise, optimise2, optimise3

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
    
    graph_data = data["graph_data"]
    graph = json_graph.node_link_graph(graph_data)

    # uncomment to get adj matrix
    # adj_matrix = nx.adjacency_matrix(graph, weight='weight').toarray()
    
    return graph, graph_name, color_set_size, degree, num_nodes, gaussian_mean, gaussian_variance



if __name__ == '__main__':

    file_path = r"C:\Projects\Heuristics for combinatorial optimisation\Heuristics-for-combinatorial-optimisation\data\graphs\test7.json"
    graph, graph_name, color_set_size, degree, num_nodes, gaussian_mean, gaussian_variance = load_graph_from_json(file_path)
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

    start_time = time.time()

    # graph, final_cost, iterations_taken, cost_data = optimise(graph, color_set_size, algo = 'reluctant')
    # graph, final_cost, iterations_taken, cost_data = optimise2(graph, color_set_size, algo = 'reluctant')
    
    fg = lambda x: x # greedy transforamtion to cost change matrix
    def fr(x):
        # Check if x is a NumPy array
        if isinstance(x, np.ndarray):
            # Vectorize for arrays
            vectorized_func = np.vectorize(lambda x: 0.0 if x == 0 else 1.0 / x)
            return vectorized_func(x)
        else:
            # Apply directly for scalars
            return 0.0 if x == 0 else 1.0 / x
    # fr = np.vectorize(lambda x: 0.0 if x == 0 else 1.0 / float(x)) # reluctant transformation to cost change matrix, np vectorisation to handle each ele individually

    graph, final_cost, iterations_taken, cost_data = optimise3(graph, color_set_size, algo_func=fr) 

    print("--- %s seconds ---" % (time.time() - start_time))

    # uncomment to visualise graph plot aft optimisation
    draw_graph(graph, pos=nx.spring_layout(graph, seed=1), 
               graph_name=graph_name, 
               iterations_taken=iterations_taken, 
               cost_data=cost_data,
               color_set_size=color_set_size, 
               degree=degree, 
               num_nodes=num_nodes, 
               gaussian_mean=gaussian_mean, 
               gaussian_variance=gaussian_variance
               )

    