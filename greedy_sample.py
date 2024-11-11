# quick implementation of greedy coloring, gtc 11/11/2024
import numpy as np
import networkx as nx
import heapdict
import time

def move_cost(G, g, i, K):
    ans = np.zeros(K)
    for r in range(K):
        ans[r] = np.sum([G.edges[(i,j)]['weight']*(-1 if r==g[j] else 1) for j in G.neighbors(i)])
    return ans - ans[g[i]]

def greedy_color(G, g, K):
    C = np.array([move_cost(G, g, i, K) for i in range(n)])
    m = heapdict.heapdict() 
    for i in G:
        m[i] = np.min(C[i])
    while True:
        i, C_i = m.popitem()
        if C_i >= 0.0:
            break
        g[i] = np.argmin(C[i])
        C[i] = move_cost(G, g, i, K)
        m[i] = 0.0
        for j in G.neighbors(i):
            C[j] = move_cost(G, g, j, K)
            m[j] = np.min(C[j])
    return g

n = 10000
G = nx.random_regular_graph(20, n)
for i,j in G.edges():
    G.edges[(i,j)]["weight"] = np.random.normal()

K = 4
t = -time.time()
g = greedy_color(G, np.random.randint(K, size=n), K)
t += time.time()
print(f"time take: {t} seconds")

