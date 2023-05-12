import numpy as np
from .gradientdescent import GradientDescent

# DONE: Implement Adam optimizer
class Adam:
    def __init__(self, layers_list, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.layers = layers_list
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.V = {}
        self.S = {}
        for layer_idx, layer in enumerate(layers_list):
            # DONE: Initialize V and S for each layer (v and s are lists of zeros with the same shape as the parameters)
            v = [np.zeros_like(p) for p in layer.parameters]
            s = [np.zeros_like(p) for p in layer.parameters]
            self.V[layer_idx] = v
            self.S[layer_idx] = s
        
    def update(self, grads, name, epoch):
        layer = self.layers[None]
        params = []
        for param_idx in range(len(grads)):
            # DONE: Implement Adam update            
            self.V[name][param_idx] = self.beta1 * self.V[name][param_idx] + (1 - self.beta1) * grads[param_idx]
            self.S[name][param_idx] = self.beta2 * self.S[name][param_idx] + (1 - self.beta2) * np.square(grads[param_idx])
            V_corrected = self.V[name][param_idx] / (1 - np.power(self.beta1, epoch))
            S_corrected = self.S[name][param_idx] / (1 - np.power(self.beta2, epoch))
            params.append(params[param_idx] - self.learning_rate * V_corrected / (np.sqrt(S_corrected) + self.epsilon))

        return params