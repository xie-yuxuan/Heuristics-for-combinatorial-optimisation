random_prob = 0

graph_name = "SBM(10000, 2, t70)"

if random_prob is not None:
    graph_name = graph_name[:-1] + f", {random_prob})"

print(graph_name)