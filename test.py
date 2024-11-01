import numpy as np

cost_change_matrix = np.zeros((5, 4))
cost_change_matrix[0][0] = 0.21

fg = lambda x: x
fr = np.vectorize(lambda x: 0 if x == 0 else 1 / x) # numpy vectorisation to handle each ele individually
def opt(func):

    return func(cost_change_matrix)

print(opt(fr))