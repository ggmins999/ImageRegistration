import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(-x))


training_inputs = np.array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
training_outputs = np.array([[1,0,0,1]]).T
np.random.seed(1)
synaptic_weights = 1 * np.random.random((3,1)) - 1

print('Random starting synaptic weights: ')
print(synaptic_weights)


