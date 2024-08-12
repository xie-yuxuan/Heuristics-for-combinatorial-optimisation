from utils import calc_cost, calc_delta_cost

def naive_greedy(graph, color_set_size, iterations):
    # Initialise cost and color set
    cur_cost = calc_cost(graph)
    color_set = list(range(color_set_size))
    itertions_taken = 0
    
    # Naive greedy algo, for loop based on fixed no. of iterations
    for i in range(iterations):
        # Determine choice combination by largest cost reduction
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

    return graph, cur_cost, itertions_taken

def animate_naive_greedy(graph, color_set_size, iterations):
    cur_cost = calc_cost(graph)
    color_set = list(range(color_set_size))
    iterations_taken = 0

    yield graph, cur_cost, iterations_taken # Yield initial state

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

        yield graph, cur_cost, iterations_taken # Yield updated state

def reluctant(graph, color_set_size, iterations):
    pass