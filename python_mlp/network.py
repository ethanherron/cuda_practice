# network.py

import numpy as np

from engine import Variable
from utils import ReLU

class Linear:
    def __init__(self, in_features, out_features, activation=None):
        self.W = Variable(np.random.randn(in_features, out_features) * np.sqrt(2. / in_features))
        self.b = Variable(np.zeros(out_features))
        self.activation = activation
        
    def __call__(self, x):
        linear_output = (x @ self.W) + self.b
        if self.activation:
            return self.activation.apply(linear_output)
        return linear_output

    def parameters(self):
        return self.W, self.b
    
class MLP:
    def __init__(self, in_features, out_features, hidden_features, num_layers):
        self.mlp = [Linear(in_features, hidden_features, ReLU)]
        for _ in range(num_layers):
            self.mlp.append(Linear(hidden_features, hidden_features, ReLU))
        self.mlp.append(Linear(hidden_features, out_features))
        
    def __call__(self, x):
        for layer in self.mlp:
            x = layer(x)
        return x
    
    def parameters(self):
        params = []
        for layer in self.mlp:
            params.extend(layer.parameters())
        return params
        
        