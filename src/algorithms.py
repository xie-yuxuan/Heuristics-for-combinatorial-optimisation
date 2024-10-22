import heapq

from utils import calc_cost, calc_delta_cost

def optimise(graph, color_set_size, algo):
    # Initialise cost, iteration count
    cur_cost = calc_cost(graph)
    iterations_taken = 0

    sorted_cost_list = []

    for node in graph.nodes:
        current_color = graph.nodes[node]['color']
        for color in range(color_set_size):
            if color != current_color: # to only consider other color choices
                delta_cost = calc_delta_cost(graph, node, current_color, color)
                heapq.heappush(sorted_cost_list, (-delta_cost, node, color)) # list of cost change, first choice for greedy

    while sorted_cost_list:
        i = 0
        delta_cost, node, new_color = heapq.heappop(sorted_cost_list)
        delta_cost = -delta_cost # change back to +ve, represent cost reduction

        if delta_cost <= 0:
            # reach convergence, no more choice that will res in cost reduction
            break

        if i == 5:
            # to avoid infinite loop, for now
            break
        
        # recolor
        current_color = graph.nodes[node]['color']
        graph.nodes[node]['color'] = new_color
        cur_cost -= delta_cost
        iterations_taken += 1

        # TODO
        # recompute cost reductions for itself when changing to other colors and update list, cost reduction computes for all edges
        # recompute cost reductions for neighbour and update list, cost reduction computes for only one edge
        
        # use sortedlist instead of heapq to allow discarding, at any one time length of list is the same

        # put greedy and reluctant in same fn, if statements

    return sorted_cost_list

def naive_greedy(graph, color_set_size, iterations):
    # Initialise cost and color set
    cur_cost = calc_cost(graph)
    color_set = list(range(color_set_size))
    iterations_taken = 0

    # Collect data for cost plot
    cost_data = {
        'iterations': [0],
        'costs': [calc_cost(graph)]
    }
    
    # Naive greedy algo, for loop based on fixed no. of iterations
    for i in range(iterations):
        # Determine choice combination by largest cost reduction
        vertex_choice = None
        color_choice = None
        max_cost_reduction = 0

        for vertex in graph.nodes: # graph.nodes is a list of all nodes
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

        cost_data['iterations'].append(iterations_taken)
        cost_data['costs'].append(cur_cost)

    return graph, cur_cost, iterations_taken, (cost_data['iterations'], cost_data['costs'])

def animate_naive_greedy(graph, color_set_size, iterations):
    cur_cost = calc_cost(graph)
    color_set = list(range(color_set_size))
    iterations_taken = 0
    recolored_node = None

    yield graph, cur_cost, iterations_taken, None # Yield initial state

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
        recolored_node = vertex_choice

        iterations_taken = i + 1

        yield graph, cur_cost, iterations_taken, recolored_node
    yield graph, cur_cost, iterations_taken, None

def naive_reluctant(graph, color_set_size, iterations):
    # Initialize cost and color set
    cur_cost = calc_cost(graph)
    color_set = list(range(color_set_size))
    iterations_taken = 0

    # Collect data for cost plot
    cost_data = {
        'iterations': [0],
        'costs': [calc_cost(graph)]
    }

    # Naive reluctant algo, for loop based on fixed no. of iterations
    for i in range(iterations):
        # Determine choice combination by smallest positive cost reduction
        vertex_choice = None
        color_choice = None
        min_cost_reduction = float('inf')  # Start with an infinitely large value

        for vertex in graph.nodes:  # graph.nodes is a list of all nodes
            ori_color = graph.nodes[vertex]['color']

            for color in color_set:
                delta_cost = calc_delta_cost(graph, vertex, ori_color, color)

                # We are looking for the smallest positive reduction
                if 0 < delta_cost < min_cost_reduction:
                    min_cost_reduction = delta_cost
                    vertex_choice = vertex
                    color_choice = color

        # If no reduction can be found, stop the algorithm
        if min_cost_reduction == float('inf'):
            break

        # Recoloring
        graph.nodes[vertex_choice]['color'] = color_choice
        cur_cost -= min_cost_reduction
        iterations_taken = i + 1

        cost_data['iterations'].append(iterations_taken)
        cost_data['costs'].append(cur_cost)

    return graph, cur_cost, iterations_taken, (cost_data['iterations'], cost_data['costs'])

def animate_naive_reluctant(graph, color_set_size, iterations):
    cur_cost = calc_cost(graph)
    color_set = list(range(color_set_size))
    iterations_taken = 0
    recolored_node = None

    yield graph, cur_cost, iterations_taken, None # Yield initial state

    for i in range(iterations):
        vertex_choice = None
        color_choice = None
        min_cost_reduction = float('inf')

        for vertex in graph.nodes:
            ori_color = graph.nodes[vertex]['color']

            for color in color_set:
                delta_cost = calc_delta_cost(graph, vertex, ori_color, color)

                if 0 < delta_cost < min_cost_reduction:
                    min_cost_reduction = delta_cost
                    vertex_choice = vertex
                    color_choice = color

        if min_cost_reduction == float('inf'):
            break
        
        # Recoloring 
        graph.nodes[vertex_choice]['color'] = color_choice
        cur_cost -= min_cost_reduction
        recolored_node = vertex_choice

        iterations_taken = i + 1

        yield graph, cur_cost, iterations_taken, recolored_node
    yield graph, cur_cost, iterations_taken, None