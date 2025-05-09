import numpy as np
from sortedcontainers import SortedList, SortedSet
from collections import defaultdict
import json
import os
import networkx as nx
import time
import random
import copy
from networkx.readwrite import json_graph

from utils import calc_cost, calc_delta_cost, calc_delta_cost_edge, calc_log_likelihood, compute_w, bisect

class SBMState:
    def __init__(self, graph, num_groups, w): # take w as an arugment, play with diff w
        self.graph = graph
        self.num_groups = num_groups

        """
        Linear SBM optimisation function 
        Initialise and maintains matrices C and N
        C is a matrix of heaps for the first term in log likelihood equation
        N is a matrix of values for the second term in log likelihood equation
        In each iteration, make a matrix from C's first element if greedy then add to N, the largest ele in the resulting matrix tells you the move to make
        If reluctant, apply algo function to heaps so first element becomes the target for reluctant
        Update n, update N (specific elements), update C (specific elements of specific heaps), update g 
        """

        self.g = np.array([graph.nodes[node]['color'] for node in graph.nodes])
        self.n = np.zeros(num_groups)
        self.m = np.zeros((num_groups, num_groups))
        for node in graph.nodes():
            self.n[self.g[node]] += 1
        for u, v in graph.edges():
            self.m[self.g[v], self.g[u]] = self.m[self.g[u], self.g[v]] = self.m[self.g[u], self.g[v]] + 1

        self.w = w.copy()

        self.log_likelihood = calc_log_likelihood(self.n, self.m, self.w)
        self.log_likelihood_data = [self.log_likelihood]
        
        self.N = np.zeros((num_groups, num_groups))
        self.C = np.empty((num_groups, num_groups), dtype=object)

        for (r, s) in [(r, s) for r in range(num_groups) for s in range(num_groups)]:
        # for (r, s) in [(r, s) for r in range(num_groups) for s in range(num_groups) if r != s]:
            self.C[r, s] = SortedSet() # not sure what complexity this is
            
            n_after = self.n.copy()
            n_after[r] -= 1
            n_after[s] += 1

            # 2nd term cost difference
            self.N[r, s] = np.nansum(
                np.triu(
                    ((np.outer(n_after, n_after) - np.diag(0.5 * n_after * (n_after + 1))) -  (np.outer(self.n, self.n) - np.diag(0.5 * self.n * (self.n + 1)))) * np.log(1 - self.w)
                )
            )
            if abs(self.N[r, s]) < 1e-13:
                self.N[r, s] = 0

        self.cost_change_matrix = np.zeros((len(graph.nodes), num_groups))
        self._update_cost_change_matrix_and_C() 


    def _update_cost_change_matrix_and_C(self, nodes_to_update=None):
        nodes_to_update = nodes_to_update or self.graph.nodes()
        
        for node in nodes_to_update:
            current_color = self.g[node]

            self.cost_change_matrix[node, current_color] = 0
            
            for color in range(self.num_groups):
                if color != current_color:
                    m_after = self.m.copy()
                    original_color = self.g[node]
                    self.g[node] = color
                    
                    for neighbor in self.graph.neighbors(node):
                        m_after[current_color, self.g[neighbor]] = m_after[self.g[neighbor], current_color] = m_after[self.g[neighbor], current_color] - 1
                        m_after[color, self.g[neighbor]] = m_after[self.g[neighbor], color] = m_after[self.g[neighbor], color] + 1    
                    
                    self.g[node] = original_color
                    
                    self.cost_change_matrix[node, color] = np.nansum(
                        np.triu((m_after - self.m) * np.log(self.w / (1 - self.w)))
                    )
                    if abs(self.cost_change_matrix[node, color]) < 1e-13:
                        self.cost_change_matrix[node, color] = 0
                    
                    self.C[current_color, color].add((self.cost_change_matrix[node, color], node))

    def find_best_move(self, algo_func, random_prob):
        if algo_func == "greedy":
            C_processed = np.array([
                [0 if len(cell) == 0 else cell[-1][0] for cell in row] for row in self.C
            ], dtype=float)
            
            log_likelihood_matrix = C_processed + self.N

            group_change = bef, aft = np.unravel_index(np.argmax(log_likelihood_matrix, axis=None), log_likelihood_matrix.shape)

            log_likelihood_change = log_likelihood_matrix[group_change]

            if log_likelihood_change <= 0:
                return None, None, None
            node_to_move = self.C[bef, aft][-1][-1]
            return node_to_move, aft, log_likelihood_change
            
        elif algo_func == "greedy_random":
            if np.random.rand() < random_prob:  
                valid_nodes = np.arange(len(self.g))  # All nodes are valid choices
                node_to_move = np.random.choice(valid_nodes)  # Pick a random node
                
                # Get current color of the node
                current_color = self.g[node_to_move]
                
                # Pick a different color (any color that is not the current color)
                available_colors = np.delete(np.arange(self.num_groups), current_color)
                aft = np.random.choice(available_colors)

                # log_likelihood_change = next((ll for ll, n in first_term_heap if n == node_to_move), None) + second_term
                log_likelihood_change = self.cost_change_matrix[node_to_move][aft] + self.N[current_color][aft]

                return node_to_move, aft, log_likelihood_change

            else:
                C_processed = np.array([
                    [0 if len(cell) == 0 else cell[-1][0] for cell in row] for row in self.C
                ], dtype=float)
                
                log_likelihood_matrix = C_processed + self.N

                group_change = bef, aft = np.unravel_index(np.argmax(log_likelihood_matrix, axis=None), log_likelihood_matrix.shape)

                log_likelihood_change = log_likelihood_matrix[group_change]

                if log_likelihood_change <= 0:
                    return None, None, None
                node_to_move = self.C[bef, aft][-1][-1]
                return node_to_move, aft, log_likelihood_change

        
        elif algo_func == "reluctant":
        
            # log_likelihood_matrix = np.empty_like(self.C, dtype=object)
            log_likelihood_matrix = np.full_like(self.C, np.nan, dtype=float)

            # chosen_nodes = np.empty_like(self.C, dtype=object)
            chosen_nodes = np.full_like(self.C, np.nan, dtype=object)

            for (r, s) in [(r, s) for r in range(self.num_groups) for s in range(self.num_groups) if r != s]:   
                chosen_tuple = bisect(self.C[r, s], self.N[r, s])
                result = chosen_tuple[0] + self.N[r, s] if chosen_tuple else np.nan # log likelihood matrix is after adding N
                # need to threshold this matrix if element is less than 1e-13
                log_likelihood_matrix[r, s] = result if result is np.nan or result >= 1e-13 else np.nan
                chosen_nodes[r, s] = chosen_tuple[1] if chosen_tuple else np.nan 
            # print(self.C)
            # print(self.N)
            # print(log_likelihood_matrix)
            # print(chosen_nodes)

            if np.all(log_likelihood_matrix == np.nan):
                return None, None, None
            
            # Convert to NumPy array, replacing None with np.nan
            arr = np.array(log_likelihood_matrix, dtype=float)
            # arr[arr <= 0] = np.nan  # Set non-positive values to NaN
            
            # Find the minimum positive value
            min_val = np.nanmin(arr) if np.any(np.isfinite(arr)) else None
            
            if min_val is not None:
                # Get the index of the minimum positive value
                group_change = bef, aft = np.argwhere(arr == min_val)[0]
                # print(group_change)
                node_to_move = chosen_nodes[bef, aft]
                log_likelihood_change = log_likelihood_matrix[bef, aft]
                return node_to_move, aft, log_likelihood_change
            return None, None, None
        
        elif algo_func == "reluctant_random":
            if np.random.rand() < random_prob:  
                valid_nodes = np.arange(len(self.g))  # All nodes are valid choices
                node_to_move = np.random.choice(valid_nodes)  # Pick a random node
                
                # Get current color of the node
                current_color = self.g[node_to_move]
                
                # Pick a different color (any color that is not the current color)
                available_colors = np.delete(np.arange(self.num_groups), current_color)
                aft = np.random.choice(available_colors)
                
                # log_likelihood_change = next((ll for ll, n in first_term_heap if n == node_to_move), None) + second_term
                log_likelihood_change = self.cost_change_matrix[node_to_move][aft] + self.N[current_color][aft]

                return node_to_move, aft, log_likelihood_change

            else:
                log_likelihood_matrix = np.full_like(self.C, np.nan, dtype=float)

                # chosen_nodes = np.empty_like(self.C, dtype=object)
                chosen_nodes = np.full_like(self.C, np.nan, dtype=object)

                for (r, s) in [(r, s) for r in range(self.num_groups) for s in range(self.num_groups) if r != s]:   
                    chosen_tuple = bisect(self.C[r, s], self.N[r, s])
                    result = chosen_tuple[0] + self.N[r, s] if chosen_tuple else np.nan # log likelihood matrix is after adding N
                    # need to threshold this matrix if element is less than 1e-13
                    log_likelihood_matrix[r, s] = result if result is np.nan or result >= 1e-13 else np.nan
                    chosen_nodes[r, s] = chosen_tuple[1] if chosen_tuple else np.nan 
                # print(self.C)
                # print(self.N)
                # print(log_likelihood_matrix)
                # print(chosen_nodes)

                if np.all(log_likelihood_matrix == np.nan):
                    return None, None, None
                
                # Convert to NumPy array, replacing None with np.nan
                arr = np.array(log_likelihood_matrix, dtype=float)
                # arr[arr <= 0] = np.nan  # Set non-positive values to NaN
                
                # Find the minimum positive value
                min_val = np.nanmin(arr) if np.any(np.isfinite(arr)) else None
                
                if min_val is not None:
                    # Get the index of the minimum positive value
                    group_change = bef, aft = np.argwhere(arr == min_val)[0]
                    # print(group_change)
                    node_to_move = chosen_nodes[bef, aft]
                    log_likelihood_change = log_likelihood_matrix[bef, aft]
                    return node_to_move, aft, log_likelihood_change
                return None, None, None
            
    def change_color(self, node, new_color):
        old_color = self.g[node]
        self.n[old_color] -= 1
        self.n[new_color] += 1
        self.g[node] = new_color
        
        for neighbor in self.graph.neighbors(node):
            self.m[old_color, self.g[neighbor]] = self.m[self.g[neighbor], old_color] = self.m[self.g[neighbor], old_color] - 1
            self.m[new_color, self.g[neighbor]] = self.m[self.g[neighbor], new_color] = self.m[self.g[neighbor], new_color] + 1
    
        # update elements in N
        for (r, s) in [(r, s) for r in range(self.num_groups) for s in range(self.num_groups) if r != s]:   
            n_after = self.n.copy()
            n_after[r] -= 1
            n_after[s] += 1

            # 2nd term cost difference
            self.N[r, s] = np.nansum(
                np.triu(
                    ((np.outer(n_after, n_after) - np.diag(0.5 * n_after * (n_after + 1))) -  (np.outer(self.n, self.n) - np.diag(0.5 * self.n * (self.n + 1)))) * np.log(1 - self.w)
                )
            )
            if abs(self.N[r, s]) < 1e-13:
                self.N[r, s] = 0

        affected_nodes = [node] + list(self.graph.neighbors(node))
        # remove node and its neighbors from any heaps in C
        for (r, s) in [(r, s) for r in range(self.num_groups) for s in range(self.num_groups) if r != s]: 
            for node_to_remove in affected_nodes:
                self.C[r, s].discard((self.cost_change_matrix[node_to_remove, s], node_to_remove))

        self._update_cost_change_matrix_and_C(affected_nodes)

    def optimise(self, algo_func, random_prob, max_iterations):
        iteration = 0
        # changes = []
        while True:
        # for i in range(0):

            node_to_move, new_color, log_likelihood_change = self.find_best_move(algo_func, random_prob)
            # if node_to_move != None:
            #     changes.append([node_to_move, new_color])

            if log_likelihood_change is None:
                # self.log_likelihood_data[0].append(iteration)
                # self.log_likelihood_data[1].append(self.log_likelihood)
                break
            self.change_color(node_to_move, new_color)
            iteration += 1
            # if iteration % 1000 == 0:
            #     print(iteration)
            self.log_likelihood += log_likelihood_change
            # self.log_likelihood_data[0].append(iteration)
            self.log_likelihood_data.append(self.log_likelihood)
        
        # loop through g and then recolor all nodes
        for node, color in enumerate(self.g):
            self.graph.nodes[node]['color'] = color

        # final_LL = calc_log_likelihood(self.n, self.m, self.w)

        return self.graph, self.log_likelihood_data, self.w, self.g

def optimise_sbm3(graph, num_groups, group_mode, algo_func):
    """
    EM-style SBM optimisation: 
    1. Update w initially
    2. Optimize group membership until convergence without updating w
    3. Update w
    4. Repeat steps 2-3 until w update does not improve log-likelihood
    """
    g = np.array([graph.nodes[node]['color'] for node in graph.nodes])
    n, m = np.zeros(num_groups), np.zeros((num_groups, num_groups))
    
    for node in graph.nodes():
        n[g[node]] += 1
    for u, v in graph.edges():
        m[g[v], g[u]] = m[g[u], g[v]] = m[g[u], g[v]] + 1
    
    w = compute_w(n, m) # w0, TODO: should this be compute or educated guess
    log_likelihood = calc_log_likelihood(n, m, w)
    log_likelihood_data = [[0], [log_likelihood]]
    iteration = 0
    
    while True:
        prev_log_likelihood = log_likelihood
        while True:
            best_increase = 0 if algo_func == "greedy" else float('inf')
            best_node, best_color = None, None
            
            for node in graph.nodes:
                current_color = graph.nodes[node]['color']
                for color in range(num_groups):
                    if color == current_color:
                        continue
                    
                    temp_n, temp_m, temp_g = n.copy(), m.copy(), g.copy()
                    graph.nodes[node]['color'] = color
                    
                    temp_n[current_color] -= 1
                    temp_n[color] += 1
                    temp_g[node] = color
                    
                    for neighbor in graph.neighbors(node):
                        temp_m[current_color, temp_g[neighbor]] = temp_m[temp_g[neighbor], current_color] = temp_m[temp_g[neighbor], current_color] - 1
                        temp_m[color, temp_g[neighbor]] = temp_m[temp_g[neighbor], color] = temp_m[temp_g[neighbor], color] + 1
                    
                    temp_log_likelihood = calc_log_likelihood(temp_n, temp_m, w)
                    increase = temp_log_likelihood - log_likelihood
                    
                    if algo_func == "greedy" and increase > 1e-13 and (increase > best_increase):
                        best_increase, best_node, best_color = increase, node, color
                    elif algo_func == "reluctant" and increase > 1e-13 and (increase < best_increase):
                        best_increase, best_node, best_color = increase, node, color
                    
                    graph.nodes[node]['color'] = current_color
            
            if best_node is None:
                break
            
            r = graph.nodes[best_node]['color']
            graph.nodes[best_node]['color'] = best_color
            n[r] -= 1
            n[best_color] += 1
            g[best_node] = best_color
            
            for neighbor in graph.neighbors(best_node):
                m[r, g[neighbor]] = m[g[neighbor], r] = m[g[neighbor], r] - 1
                m[best_color, g[neighbor]] = m[g[neighbor], best_color] = m[g[neighbor], best_color] + 1
            
            log_likelihood = calc_log_likelihood(n, m, w)
            iteration += 1
            log_likelihood_data[0].append(iteration)
            log_likelihood_data[1].append(log_likelihood)
        
        w_new = compute_w(n, m)
        new_log_likelihood = calc_log_likelihood(n, m, w_new)
        
        if abs(new_log_likelihood - prev_log_likelihood) <= 1e-13:
            print(f"Terminating at iteration {iteration}: No valid moves found.")
            break
        
        w = w_new
        log_likelihood = new_log_likelihood
        iteration += 1
        log_likelihood_data[0].append(iteration)
        log_likelihood_data[1].append(log_likelihood)
    
    return graph, log_likelihood_data, w



def optimise_sbm2(graph, num_groups, group_mode, algo_func):
    """
    sbm optimisation that updates g in each iteration, w is kept as ground truth / educated guess
    """
    # compute initial w, symmetric matrix of edge probabilities
    # initialise global n, m, g
    # n is 1D array that stores the number of nodes in each group
    # m is 2D array that stores the number of edges between groups
    # g is group membership of each node
    g = np.array([graph.nodes[node]['color'] for node in graph.nodes])
    n, m = np.zeros(num_groups), np.zeros((num_groups, num_groups))

    for node in graph.nodes():
        n[g[node]] += 1 # increment group count for each group

    for u, v in graph.edges():
        # increment edge count between groups
        # ensures m is symmetric
        m[g[v], g[u]] = m[g[u], g[v]] = m[g[u], g[v]] + 1
    
    # w0 is ground truth / educated guess
    # Generate the w matrix (edge probabilities)
    w = np.zeros((num_groups, num_groups))

    if group_mode == "association":
        w += 1  # Small baseline for non-diagonal elements
        np.fill_diagonal(w, 9)  # Large diagonal elements
    elif group_mode == "bipartite":
        w += 9  # Large baseline for non-diagonal elements
        np.fill_diagonal(w, 1)  # Small diagonal elements
    elif group_mode == "core-periphery":
        w += 9  # Large baseline
        w[0, :] = 1  # Small first row (loners have low connections to all groups)
        w[:, 0] = 1  # Small first column (low connections to loners)
        w[0, 0] = 1  # loners have low self-connections

    w /= len(graph.nodes)
    
    # compute inital log_likelihood
    log_likelihood = calc_log_likelihood(n, m, w)
    

    # initial likelihood data = [[iteration count],[log likelihood at that iteration]] which is a list of list
    log_likelihood_data = [[0], [log_likelihood]]

    iteration = 0

    # for iteration in range(1):
    while True:
        best_increase = 0 if algo_func == "greedy" else float('inf')
        best_node, best_color = None, None

        # Iterate through all nodes and possible colors
        for node in graph.nodes:
            current_color = graph.nodes[node]['color']

            for color in range(num_groups):
                if color == current_color:
                    continue

                temp_m = m.copy()
                temp_n = n.copy()
                temp_g = g.copy()

                # Temporarily recolor the node
                graph.nodes[node]['color'] = color

                # temp update temp_m and temp_n
                temp_n[current_color] -= 1
                temp_n[color] += 1
                temp_g[node] = color

                for neighbor in graph.neighbors(node):
                    temp_m[current_color, temp_g[neighbor]] = temp_m[temp_g[neighbor], current_color] = temp_m[temp_g[neighbor], current_color] - 1
                    temp_m[color, temp_g[neighbor]] = temp_m[temp_g[neighbor], color] = temp_m[temp_g[neighbor], color] + 1
                
                # Recompute w and log-likelihood
                # print("still working")
                # functionn taht will update m and n here efficiently

                # temp_w = compute_w(temp_n, temp_m)
                
                temp_log_likelihood = calc_log_likelihood(temp_n, temp_m, w)

                # Check for the best increase
                increase = temp_log_likelihood - log_likelihood

                # Greedy: Maximize positive increase
                if algo_func == "greedy" and increase > 1e-13 and (increase > best_increase): # set some threshold for increase
                    best_increase, best_node, best_color = increase, node, color
                # Reluctant: Minimize positive increase
                elif algo_func == "reluctant" and increase > 1e-13 and (increase < best_increase): # set some threshold for increase
                    best_increase, best_node, best_color = increase, node, color

                # Revert the change
                graph.nodes[node]['color'] = current_color

        # If no improvement, terminate
        if best_node is None or best_color is None:
            print(f"Terminating at iteration {iteration}: No valid moves found.")
            break

        r = graph.nodes[best_node]['color']

        # Apply the best change
        # print(best_node, r, best_color)
        graph.nodes[best_node]['color'] = best_color

        # update n, m, g
        n[r] -= 1
        n[best_color] += 1
        g[best_node] = best_color

        for neighbor in graph.neighbors(best_node):
            m[r, g[neighbor]] = m[g[neighbor], r] = m[g[neighbor], r] - 1
            m[best_color, g[neighbor]] = m[g[neighbor], best_color] = m[g[neighbor], best_color] + 1

        # w = compute_w(n, m)
        log_likelihood = calc_log_likelihood(n, m, w)
        iteration += 1
        print(f"iteration: {iteration}, log_likelihood: {log_likelihood}")
        log_likelihood_data[0].append(iteration)
        log_likelihood_data[1].append(log_likelihood)

    return graph, log_likelihood_data, w


def optimise_sbm(graph, num_groups, group_mode, algo_func):
    """
    sbm optimisation that updates g and w in each iteration 
    """
    # compute initial w, symmetric matrix of edge probabilities
    # initialise global n, m, g
    # n is 1D array that stores the number of nodes in each group
    # m is 2D array that stores the number of edges between groups
    # g is group membership of each node
    g = np.array([graph.nodes[node]['color'] for node in graph.nodes])
    n, m = np.zeros(num_groups), np.zeros((num_groups, num_groups))

    for node in graph.nodes():
        n[g[node]] += 1 # increment group count for each group

    for u, v in graph.edges():
        # increment edge count between groups
        # ensures m is symmetric
        m[g[v], g[u]] = m[g[u], g[v]] = m[g[u], g[v]] + 1
    
    # w0, or start with an educated guess
    w = compute_w(n, m)
    
    # compute inital log_likelihood
    log_likelihood = calc_log_likelihood(n, m, w)
    

    # initial likelihood data = [[iteration count],[log likelihood at that iteration]] which is a list of list
    log_likelihood_data = [[0], [log_likelihood]]

    iteration = 0

    # for iteration in range(1, 100):
    while True:
        best_increase = 0 if algo_func == "greedy" else float('inf')
        best_node, best_color = None, None

        # Iterate through all nodes and possible colors
        for node in graph.nodes:
            current_color = graph.nodes[node]['color']

            for color in range(num_groups):
                if color == current_color:
                    continue

                temp_m = m.copy()
                temp_n = n.copy()
                temp_g = g.copy()

                # Temporarily recolor the node
                graph.nodes[node]['color'] = color

                # temp update temp_m and temp_n
                temp_n[current_color] -= 1
                temp_n[color] += 1
                temp_g[node] = color

                for neighbor in graph.neighbors(node):
                    temp_m[current_color, temp_g[neighbor]] = temp_m[temp_g[neighbor], current_color] = temp_m[temp_g[neighbor], current_color] - 1
                    temp_m[color, temp_g[neighbor]] = temp_m[temp_g[neighbor], color] = temp_m[temp_g[neighbor], color] + 1
                
                # Recompute w and log-likelihood
                # print("still working")
                # functionn taht will update m and n here efficiently

                temp_w = compute_w(temp_n, temp_m)
                
                temp_log_likelihood = calc_log_likelihood(temp_n, temp_m, temp_w)

                # Check for the best increase
                increase = temp_log_likelihood - log_likelihood

                # Greedy: Maximize positive increase
                if algo_func == "greedy" and increase > 1e-13 and (increase > best_increase): # set some threshold for increase
                    best_increase, best_node, best_color = increase, node, color
                # Reluctant: Minimize positive increase
                elif algo_func == "reluctant" and increase > 1e-13 and (increase < best_increase): # set some threshold for increase
                    best_increase, best_node, best_color = increase, node, color

                # Revert the change
                graph.nodes[node]['color'] = current_color

        # If no improvement, terminate
        if best_node is None or best_color is None:
            print(f"Terminating at iteration {iteration}: No valid moves found.")
            break

        r = graph.nodes[best_node]['color']

        # Apply the best change
        graph.nodes[best_node]['color'] = best_color

        # update n, m, g
        n[r] -= 1
        n[best_color] += 1
        g[best_node] = best_color

        for neighbor in graph.neighbors(best_node):
            m[r, g[neighbor]] = m[g[neighbor], r] = m[g[neighbor], r] - 1
            m[best_color, g[neighbor]] = m[g[neighbor], best_color] = m[g[neighbor], best_color] + 1

        w = compute_w(n, m)
        log_likelihood = calc_log_likelihood(n, m, w)
        iteration += 1
        # print(f"iteration: {iteration}, log_likelihood: {log_likelihood}")
        log_likelihood_data[0].append(iteration)
        log_likelihood_data[1].append(log_likelihood)

    return graph, log_likelihood_data, w

def optimise4(graph, color_set_size, algo_func):
    # initialise cost, iteration count
    cur_cost = calc_cost(graph)
    # print(cur_cost)
    iterations_taken = 0
    # collect data for cost plot
    cost_data = [cur_cost]

    # initialise cost matrix, row is node, col is color, each element is cost change for that node and color
    cost_change_matrix = np.zeros((len(graph.nodes), color_set_size))
    cost_change_matrix = cost_change_matrix.astype(float)
    # print(cost_matrix) 
    # print(cost_matrix.shape) # rows are nodes, cols are colors

    for node in graph.nodes:
        current_color = graph.nodes[node]['color']
        for color in range(color_set_size):
            if color != current_color: # to only consider other color choices
                delta_cost = calc_delta_cost(graph, node, current_color, color)
                cost_change_matrix[node][color] = -delta_cost

    # print(cost_change_matrix)

    # initialise a sorted list of cost change, first choice for recoloring
    sorted_cost_set = SortedSet()
    
    # find index of the most negative (minimum) value in each row, best color change choice for that node
    min_indices = np.argmin(algo_func(cost_change_matrix), axis=1)
    
    # retrieve the minimum values based on the indices
    min_values = cost_change_matrix[np.arange(cost_change_matrix.shape[0]), min_indices]
    
    # populate the sorted list
    for node, (cost_change, best_color) in enumerate(zip(algo_func(min_values), min_indices)): # pair 1st item in one iterable with 1st item of another iterable
        sorted_cost_set.add((cost_change, node, best_color))

    # print(sorted_cost_set) # no fr applied, everything good as original here
    # i = 0

    while True:
    # for x in range(3):

        delta_cost, node, new_color = sorted_cost_set[0]

        # print('recoloring')
        # print(delta_cost, node, new_color)
        delta_cost = -algo_func(delta_cost) # change back to +ve, represent cost reduction

        

        if delta_cost <= 0:
            # reach convergence, no more choice that will res in cost reduction
            break

        # recolor the node
        node_color_bef = graph.nodes[node]['color']
        graph.nodes[node]['color'] = new_color
        current_color = new_color
        cur_cost -= delta_cost
        iterations_taken += 1

        cost_data.append(cur_cost)

        # print(cur_cost)
        # print(calc_cost(graph))

        # update cost data

        
        # delete the recolored node and its neighbors from the sorted list
        nodes_to_remove = [node] + list(graph.neighbors(node)) # O(k)
        
        for removing_node in nodes_to_remove:
            min_idx = np.argmin(algo_func(cost_change_matrix[removing_node]))
            sorted_cost_set.remove((algo_func(cost_change_matrix[removing_node][min_idx]), removing_node, min_idx))

        # print('after removing', sorted_cost_set)

        # update cost matrix by looping through neighbors of the recolored node
        for neighbor in graph.neighbors(node):
            edge_weight = graph[node][neighbor].get('weight') 
            neighbor_color_bef = graph.nodes[neighbor]['color']
            
            for color in range(color_set_size): # looping over col
                # step 1: node row update, then update each col correctly, will be updated every iteration
                node_delta_update = edge_weight * (
                      int(node_color_bef == neighbor_color_bef) # int() to convert np bool to int
                    - int(new_color == neighbor_color_bef)
                )
            
                cost_change_matrix[node][color] += node_delta_update # update cost change in node row 
                if abs(cost_change_matrix[node][color]) < 1e-13:
                    cost_change_matrix[node][color] = 0          

                # step 2: neighbor row update, then update each col correctly, will only be updated once
                neighbor_delta_update = edge_weight * (
                      int(new_color == color) 
                    - int(new_color == neighbor_color_bef) 
                    - int(node_color_bef == color) 
                    + int(node_color_bef == neighbor_color_bef)
                )
                cost_change_matrix[neighbor][color] += neighbor_delta_update
                if abs(cost_change_matrix[neighbor][color]) < 1e-13:
                    cost_change_matrix[neighbor][color] = 0

        # add the best choice for the recolored node in the row of the cost change matrix to the sorted list
        recolored_row = algo_func((cost_change_matrix)[node])
        recolored_best_idx = np.argmin(recolored_row)  # index of the min value in this row for the recolored node
        sorted_cost_set.add((recolored_row[recolored_best_idx], node, recolored_best_idx))

        # add the best choice for the neighbors of the recolored node in the row of the cost change matrix to the sorted list
        for neighbor in graph.neighbors(node):
            neighbor_row = algo_func((cost_change_matrix)[neighbor])
            neighbor_best_idx = np.argmin(neighbor_row)  # index of the min value for this neighbor row
            sorted_cost_set.add((neighbor_row[neighbor_best_idx], neighbor, neighbor_best_idx))


    # print(cost_change_matrix)
    # print(sorted_cost_list)

    return graph, cur_cost, iterations_taken, cost_data
def optimise3(graph, color_set_size, algo_func):
    # initialise cost, iteration count
    cur_cost = calc_cost(graph)
    # print(cur_cost)
    iterations_taken = 0
    # collect data for cost plot
    cost_data = {
        'iterations': [0],
        'costs': [cur_cost]
    }

    # initialise cost matrix, row is node, col is color, each element is cost change for that node and color
    cost_change_matrix = np.zeros((len(graph.nodes), color_set_size))
    cost_change_matrix = cost_change_matrix.astype(float)
    # print(cost_matrix) 
    # print(cost_matrix.shape) # rows are nodes, cols are colors

    for node in graph.nodes:
        current_color = graph.nodes[node]['color']
        for color in range(color_set_size):
            if color != current_color: # to only consider other color choices
                delta_cost = calc_delta_cost(graph, node, current_color, color)
                cost_change_matrix[node][color] = -delta_cost

    # print(cost_change_matrix)

    # initialise a sorted list of cost change, first choice for recoloring
    sorted_cost_list = SortedList()
    
    # find index of the most negative (minimum) value in each row, best color change choice for that node
    min_indices = np.argmin(algo_func(cost_change_matrix), axis=1)
    
    # retrieve the minimum values based on the indices
    min_values = cost_change_matrix[np.arange(cost_change_matrix.shape[0]), min_indices]
    
    # populate the sorted list
    for node, (cost_change, best_color) in enumerate(zip(algo_func(min_values), min_indices)): # pair 1st item in one iterable with 1st item of another iterable
        sorted_cost_list.add((cost_change, node, best_color))

    # print(sorted_cost_list) # no fr applied, everything good as original here
    
    while True:
    # for x in range(1):

        delta_cost, node, new_color = sorted_cost_list[0]

        # print('recoloring')
        # print(delta_cost, node, new_color)
        delta_cost = -algo_func(delta_cost) # change back to +ve, represent cost reduction

        if delta_cost <= 0:
            # reach convergence, no more choice that will res in cost reduction
            break

        # recolor the node
        node_color_bef = graph.nodes[node]['color']
        graph.nodes[node]['color'] = new_color
        current_color = new_color
        cur_cost -= delta_cost
        iterations_taken += 1

        # print(cur_cost)
        # print(calc_cost(graph))

        # update cost data
        cost_data['iterations'].append(iterations_taken)
        cost_data['costs'].append(cur_cost)

        # update cost matrix by looping through neighbors of the recolored node
        for neighbor in graph.neighbors(node):
            edge_weight = graph[node][neighbor].get('weight') #TODO 
            neighbor_color_bef = graph.nodes[neighbor]['color']
            
            for color in range(color_set_size): # looping over col
                # step 1: node row update, then update each col correctly, will be updated every iteration
                node_delta_update = edge_weight * (
                      int(node_color_bef == neighbor_color_bef) # int() to convert np bool to int
                    - int(new_color == neighbor_color_bef)
                )
            
                cost_change_matrix[node][color] += node_delta_update # update cost change in node row 
                if abs(cost_change_matrix[node][color]) < 1e-14:
                    cost_change_matrix[node][color] = 0          

                # step 2: neighbor row update, then update each col correctly, will only be updated once
                neighbor_delta_update = edge_weight * (
                      int(new_color == color) 
                    - int(new_color == neighbor_color_bef) 
                    - int(node_color_bef == color) 
                    + int(node_color_bef == neighbor_color_bef)
                )
                cost_change_matrix[neighbor][color] += neighbor_delta_update
                if abs(cost_change_matrix[neighbor][color]) < 1e-14:
                    cost_change_matrix[neighbor][color] = 0
        
        # print(sorted_cost_list)
        
        # delete the recolored node and its neighbors from the sorted list
        nodes_to_remove = [node] + list(graph.neighbors(node)) # O(k)
        # iterate backwards when removing items to avoid indexing issues
        for entry in reversed(sorted_cost_list): #TODO O(N) SLOW, find a data structure that is sorted and associative, heaps, etc
            if entry[1] in nodes_to_remove:
                sorted_cost_list.remove(entry)

        # add the best choice for the recolored node in the row of the cost change matrix to the sorted list
        recolored_row = algo_func((cost_change_matrix)[node])
        recolored_best_idx = np.argmin(recolored_row)  # index of the min value in this row for the recolored node
        sorted_cost_list.add((recolored_row[recolored_best_idx], node, recolored_best_idx))

        # add the best choice for the neighbors of the recolored node in the row of the cost change matrix to the sorted list
        for neighbor in graph.neighbors(node):
            neighbor_row = algo_func((cost_change_matrix)[neighbor])
            neighbor_best_idx = np.argmin(neighbor_row)  # index of the min value for this neighbor row
            sorted_cost_list.add((neighbor_row[neighbor_best_idx], neighbor, neighbor_best_idx))

        # print(sorted_cost_list)

    # print(cost_change_matrix)
    # print(sorted_cost_list)

    return graph, cur_cost, iterations_taken, (cost_data['iterations'], cost_data['costs'])

def optimise2(graph, color_set_size, algo):
    # initialise cost, iteration count
    cur_cost = calc_cost(graph)
    # print(cur_cost)
    iterations_taken = 0
    # collect data for cost plot
    cost_data = {
        'iterations': [0],
        'costs': [cur_cost]
    }

    # initialise cost matrix, row is node, col is color, each element is cost change for that node and color
    cost_change_matrix = np.zeros((len(graph.nodes), color_set_size))
    # print(cost_matrix) 
    # print(cost_matrix.shape) # rows are nodes, cols are colors

    for node in graph.nodes:
        current_color = graph.nodes[node]['color']
        for color in range(color_set_size):
            if color != current_color: # to only consider other color choices
                delta_cost = calc_delta_cost(graph, node, current_color, color)
                cost_change_matrix[node][color] = -delta_cost

    # print(cost_change_matrix)


    while True:
    # for x in range(1):
        if algo == 'greedy':
            # list of cost change, first choice for greedy
            sorted_cost_list = SortedList()
            # find index of the most negative (minimum) value in each row, best color change choice for that node
            min_indices = np.argmin(cost_change_matrix, axis=1)
            
            # retrieve the minimum values based on the indices
            min_values = cost_change_matrix[np.arange(cost_change_matrix.shape[0]), min_indices]
            
            # populate the sorted list
            for node, (cost_change, best_color) in enumerate(zip(min_values, min_indices)): # pair 1st item in one iterable with 1st item of another iterable
                sorted_cost_list.add((cost_change, node, best_color))
            # print(sorted_cost_list)

            delta_cost, node, new_color = sorted_cost_list[0] # greedy choice
        elif algo == 'reluctant':
            # replace all 0s of cost change matrix with -ifty 
            # take reciprocal over all elements of the cost change matrix
            # find index of most negative number in each row of the cost change matrix (same as greedy)
            # add to sorted list 
            # pop from sorted list to get the reluctant choice, take reciprocal of it and negative to get the delta  
            
            # create a temporary reciprocal matrix for reluctant calculation
            # reciprocal_cost_change_matrix = np.where(cost_change_matrix == 0, -np.inf, 1 / cost_change_matrix)
            # reciprocal_cost_change_matrix = 1 / cost_change_matrix
            # reciprocal_cost_change_matrix = np.where(reciprocal_cost_change_matrix == np.inf, 0)

            # Take reciprocal first, infinities will appear where there were originally zeros
            reciprocal_cost_change_matrix = 1 / cost_change_matrix # warning error as I am dividing by 0

            # Replace positive and negative infinities with 0
            reciprocal_cost_change_matrix[np.isinf(reciprocal_cost_change_matrix)] = 0
            # print(cost_change_matrix)
            # print(reciprocal_cost_change_matrix)

            # list of cost change, first choice for greedy
            sorted_cost_list = SortedList()
            # find index of the most negative (minimum) value in each row, best color change choice for that node
            min_indices = np.argmin(reciprocal_cost_change_matrix, axis=1)
            # print(min_indices)
            
            # retrieve the minimum values based on the indices
            min_values = reciprocal_cost_change_matrix[np.arange(reciprocal_cost_change_matrix.shape[0]), min_indices]
            
            # populate the sorted list
            for node, (cost_change, best_color) in enumerate(zip(min_values, min_indices)): # pair 1st item in one iterable with 1st item of another iterable

                sorted_cost_list.add((cost_change, node, best_color))
            # print("sorted cost list")
            # print(sorted_cost_list)

            delta_cost, node, new_color = sorted_cost_list[0] # reluctant choice bef reciprocal of delta_cost
            delta_cost = 1/delta_cost




        # print('recoloring')
        # print(delta_cost, node, new_color)
        delta_cost = -delta_cost # change back to +ve, represent cost reduction

        if delta_cost <= 0:
            # reach convergence, no more choice that will res in cost reduction
            break

        # recolor the node
        node_color_bef = graph.nodes[node]['color']
        graph.nodes[node]['color'] = new_color
        current_color = new_color
        cur_cost -= delta_cost
        iterations_taken += 1

        # print(cur_cost)
        # print(calc_cost(graph))

        # update cost data
        cost_data['iterations'].append(iterations_taken)
        cost_data['costs'].append(cur_cost)

        # update cost matrix by looping through neighbors of the recolored node
        for neighbor in graph.neighbors(node):
            edge_weight = graph[node][neighbor].get('weight')
            neighbor_color_bef = graph.nodes[neighbor]['color']
            
            for color in range(color_set_size): # looping over col
                # step 1: node row update, then update each col correctly, will be updated every iteration
                node_delta_update = edge_weight * (
                      int(node_color_bef == neighbor_color_bef) # int() to convert np bool to int
                    - int(new_color == neighbor_color_bef)
                )
            
                cost_change_matrix[node][color] += node_delta_update # update cost change in node row 

                # step 2: neighbor row update, then update each col correctly, will only be updated once
                neighbor_delta_update = edge_weight * (
                      int(new_color == color) 
                    - int(new_color == neighbor_color_bef) 
                    - int(node_color_bef == color) 
                    + int(node_color_bef == neighbor_color_bef)
                )
                cost_change_matrix[neighbor][color] += neighbor_delta_update

                # replace very small number in the cost_change_matrix with 0
                cost_change_matrix[np.abs(cost_change_matrix) < 1e-11] = 0
        
    # print(cost_change_matrix)
    
    return graph, cur_cost, iterations_taken, (cost_data['iterations'], cost_data['costs'])

        







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

    i = 0
    
    while sorted_cost_list:
        # print(sorted_cost_list)
        
        if algo == 'greedy':
            delta_cost, node, new_color = sorted_cost_list[0] # get first entry, lookup is O(logn)
        elif algo == 'reluctant':
            # get index of last entry that is still < 0
            for i in range(len(sorted_cost_list) - 1, -1, -1): # loop from end of the list backwards, traversal O(n)
                if sorted_cost_list[i][0] < 0:
                    reluctant_choice_index = i
                    break
            else:
                reluctant_choice_index = 0
            delta_cost, node, new_color = sorted_cost_list[reluctant_choice_index]

        # print('recoloring')
        # print(delta_cost, node, new_color)
        delta_cost = -delta_cost # change back to +ve, represent cost reduction

        if delta_cost <= 0:
            # reach convergence, no more choice that will res in cost reduction
            break
        # if i == 2:
        #     break

        # recolor
        color_bef = graph.nodes[node]['color']
        graph.nodes[node]['color'] = new_color
        current_color = new_color
        cur_cost -= delta_cost
        iterations_taken += 1

        # print(cur_cost)
        # print(calc_cost(graph))

        # update cost data
        cost_data['iterations'].append(iterations_taken)
        cost_data['costs'].append(cur_cost)

        
        # update sortedlist for node itself after recoloring
        # step 1: remove old entries related to this node
        to_remove = [[delta, n, c] for [delta, n, c] in sorted_cost_list if n == node] # traversal O(n)


        
        for [delta, n, c] in to_remove:
            sorted_cost_list.remove([delta, n, c]) # remove requires entries to be present, as opposed to discard, O(logn)

        # step 2: recalculate cost reductions for itself when changing to other colors and update list    
        for color in range(color_set_size):
            if color != current_color: # to only consider other color choices
                delta_cost = calc_delta_cost(graph, node, current_color, color) 
                sorted_cost_list.add([-delta_cost, node, color]) # add is O(logn) 

        

        # update sortedlist for neighbors after recoloring, cost reduction computes for only one edge, update previous cost reduction
        # step 1: remove old entries of neighbors
        neighbor_entries = []
        neighbor_indices = []
        
        # TODO: reduce complexity here O(knc)
        for index, entry in enumerate(sorted_cost_list):
            if entry[1] in graph.neighbors(node):
                neighbor_entries.append(entry)
                neighbor_indices.append(index)
                    
        for index in sorted(neighbor_indices, reverse=True):  # remove in reverse order to avoid index issues
            sorted_cost_list.pop(index)

        # step 2: recalculate cost reductions for neighbors by only calculating impact of one edge
        for entry in neighbor_entries:
            neighbor = entry[1]
            # print('entry 0')
            # print(entry[0])
            entry[0] += calc_delta_cost_edge(graph, node, 
                                                node_color_bef = color_bef, 
                                                node_color_aft = new_color, 
                                                neighbor_node = neighbor, 
                                                neighbor_color_bef = graph.nodes[neighbor]['color'], 
                                                neighbor_color_aft = entry[2])
            # print(entry[0], neighbor, entry[2])
            sorted_cost_list.add(entry)

        i += 1


    return graph, cur_cost, iterations_taken, (cost_data['iterations'], cost_data['costs'])




# --------------------------------------------------------------

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