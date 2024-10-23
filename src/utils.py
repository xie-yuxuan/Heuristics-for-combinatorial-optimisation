import networkx as nx

"""
Utility functions including:
- calc_cost
- calc_delta_cost
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
    Calc cost reduction (delta) when a vertex is recolored, +ve means cost is reduced
    """
    delta = 0

    for neighbor in graph.neighbors(vertex):
        neighbor_color = graph.nodes[neighbor]['color']

        if ori_color == neighbor_color:
            delta += graph[vertex][neighbor].get('weight')

        if new_color == neighbor_color:
            delta -= graph[vertex][neighbor].get('weight')

    return delta

def calc_delta_cost_edge(graph, node, node_color_bef, node_color_aft, neighbor_node, neighbor_color_bef, neighbor_color_aft):

    edge_weight = graph[node][neighbor_node].get('weight')

    if node_color_bef == neighbor_color_bef and node_color_aft == neighbor_color_aft:
        return 0
    elif node_color_bef == neighbor_color_bef and node_color_aft != neighbor_color_aft:
        return edge_weight
    elif node_color_bef != neighbor_color_bef and node_color_aft == neighbor_color_aft:
        return -edge_weight
    else:
        return 0


if __name__ == '__main__':

    pass