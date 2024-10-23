from sortedcontainers import SortedList

from utils import calc_cost, calc_delta_cost, calc_delta_cost_edge

def optimise(graph, color_set_size, algo):
    # initialise cost, iteration count
    cur_cost = calc_cost(graph)
    iterations_taken = 0
    # collect data for cost plot
    cost_data = {
        'iterations': [0],
        'costs': [cur_cost]
    }

    # list of cost change, first choice for greedy
    sorted_cost_list = SortedList()

    for node in graph.nodes:
        current_color = graph.nodes[node]['color']
        for color in range(color_set_size):
            if color != current_color: # to only consider other color choices
                delta_cost = calc_delta_cost(graph, node, current_color, color)
                sorted_cost_list.add([-delta_cost, node, color]) # add is O(logn), using a list of lists to make cost of neighbors mutable

    while sorted_cost_list:
        if algo == 'greedy':
            delta_cost, node, new_color = sorted_cost_list[0] # get first entry
        elif algo == 'reluctant':
            # get index of last entry that is still < 0
            for i in range(len(sorted_cost_list) - 1, -1, -1): # loop from end of the list backwards
                if sorted_cost_list[i][0] < 0:
                    reluctant_choice_index = i
                    break
            else:
                reluctant_choice_index = 0
            delta_cost, node, new_color = sorted_cost_list[reluctant_choice_index]

        # print(delta_cost, node, new_color)
        delta_cost = -delta_cost # change back to +ve, represent cost reduction

        if delta_cost <= 0:
            # reach convergence, no more choice that will res in cost reduction
            break

        # recolor
        color_bef = graph.nodes[node]['color']
        graph.nodes[node]['color'] = new_color
        current_color = new_color
        cur_cost -= delta_cost
        iterations_taken += 1

        # update cost data
        cost_data['iterations'].append(iterations_taken)
        cost_data['costs'].append(cur_cost)

        # update sortedlist for node itself after recoloring
        # step 1: remove old entries related to this node
        to_remove = [[delta, n, c] for [delta, n, c] in sorted_cost_list if n == node] # remove requires entries to be present, as opposed to discard, O(logn)
        
        for [delta, n, c] in to_remove:
            sorted_cost_list.remove([delta, n, c])

        # step 2: recalculate cost reductions for itself when changing to other colors and update list    
        for color in range(color_set_size):
            if color != current_color: # to only consider other color choices
                delta_cost = calc_delta_cost(graph, node, current_color, color) 
                sorted_cost_list.add([-delta_cost, node, color]) # add is O(logn)

        # update sortedlist for neighbors after recoloring, cost reduction computes for only one edge, update previous cost reduction
        # step 1: remove old entries of neighbors
        neighbor_entries = []
        neighbor_indices = []
        
        for neighbor in graph.neighbors(node):
            for index, entry in enumerate(sorted_cost_list):
                if entry[1] == neighbor:
                    neighbor_entries.append(entry)
                    neighbor_indices.append(index)
                    
        for index in sorted(neighbor_indices, reverse=True):  # remove in reverse order to avoid index issues
            sorted_cost_list.pop(index)

        # step 2: recalculate cost reductions for neighbors by only calculating impact of one edge
        for entry in neighbor_entries:
            neighbor = entry[1]
            entry[0] += calc_delta_cost_edge(graph, node, 
                                                node_color_bef = color_bef, 
                                                node_color_aft = new_color, 
                                                neighbor_node = neighbor, 
                                                neighbor_color_bef = graph.nodes[neighbor]['color'], 
                                                neighbor_color_aft = entry[2])
            sorted_cost_list.add(entry)

    return graph, cur_cost, iterations_taken, (cost_data['iterations'], cost_data['costs'])

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