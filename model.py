from layers.convolution2d import Conv2D
from layers.maxpooling2d import MaxPool2D
from layers.fullyconnected import FC

from activations import Activation, get_activation

import pickle
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle

class Model:
    def __init__(self, arch, criterion, optimizer, name=None):
        """
        Initialize the model.
        args:
            arch: dictionary containing the architecture of the model
            criterion: loss 
            optimizer: optimizer
            name: name of the model
        """
        if name is None:
            self.model = arch
            self.criterion = criterion
            self.optimizer = optimizer
            self.layers_names = list(arch.keys())
        else:
            self.model, self.criterion, self.optimizer, self.layers_names = self.load_model(name)
    
    def is_layer(self, layer):
        """
        Check if the layer is a layer.
        args:
            layer: layer to be checked
        returns:
            True if the layer is a layer, False otherwise
        """
        # DONE: Implement check if the layer is a layer
        if isinstance(layer, (Conv2D, MaxPool2D, FC)):
            return True
        else:
            return False

    def is_activation(self, layer):
        """
        Check if the layer is an activation function.
        args:
            layer: layer to be checked
        returns:
            True if the layer is an activation function, False otherwise
        """
        # DONE: Implement check if the layer is an activation
        if isinstance(layer, Activation):
            return True
        else:
            return False

    def forward(self, x):
        """
        Forward pass through the model.
        args:
            x: input to the model
        returns:
            output of the model
        """
        tmp = []
        A = x
        # DONE: Implement forward pass through the model
        # NOTICE: we have a pattern of layers and activations
        for l in range(len(self.layers_names)):
            layer = self.model[self.layers_names[l]]
            if self.is_layer(layer):
                Z = layer.forward(A)
                tmp.append(Z.copy())
                A = get_activation(self.model[self.layers_names[l+1]])(Z)
                tmp.append(A.copy())
            elif self.is_activation(layer):
                A = layer.forward(A)
                tmp.append(A.copy())
        return tmp
    
    def backward(self, dAL, tmp, x):
        """
        Backward pass through the model.
        args:
            dAL: derivative of the cost with respect to the output of the model
            tmp: list containing the intermediate values of Z and A
            x: input to the model
        returns:
            gradients of the model
        """
        dA = dAL
        grads = {}
        # DONE: Implement backward pass through the model
        # NOTICE: we have a pattern of layers and activations
        # for from the end to the beginning of the tmp list
        for l in range(len(tmp), 0, -1):
            if l > 2:
                Z, A = tmp[l - 1], tmp[l - 2]
            else:
                Z, A = tmp[l - 1], x
            dZ = get_activation(self.model[self.layers_names[l-1]], derivative=True)(dA, Z)
            dA, grad = self.model[self.layers_names[l-1]].backward(dZ, A)
            grads[self.layers_names[l-1]] = grad
        return grads

    def update(self, grads):
        """
        Update the model.
        args:
            grads: gradients of the model
        """
        for layer_name in self.layers_names:
            layer = self.model[layer_name]
            # hint check if the layer is a layer and also is not a maxpooling layer
            if self.is_layer(layer) and not isinstance(layer, MaxPool2D):
                layer.update(grads[layer_name], self.optimizer)

    def one_epoch(self, x, y, batch_size):
        """
        One epoch of training.
        args:
            x: input to the model
            y: labels
            batch_size: batch size
        returns:
            loss
        """
        # DONE: Implement one epoch of training
        order = self.shuffle(x.shape[0], True)
        cost = 0
        for b in range(0, x.shape[0], batch_size):
            bx, by = self.batch(x, y, batch_size, b, order)
            tmp = self.forward(bx)
            AL = tmp[-1]
            cost += self.criterion(AL, by)
            dAL = self.criterion.derivative(AL, by)
            grads = self.backward(dAL, tmp, bx)
            self.update(grads)
        return cost / (x.shape[0] // batch_size)

    def save(self, name):
        """
        Save the model.
        args:
            name: name of the model
        """
        with open(name, 'wb') as f:
            pickle.dump((self.model, self.criterion, self.optimizer, self.layers_names), f)
        
    def load_model(self, name):
        """
        Load the model.
        args:
            name: name of the model
        returns:
            model, criterion, optimizer, layers_names
        """
        with open(name, 'rb') as f:
            return pickle.load(f)
        
    def shuffle(self, m, shuffling):
        order = list(range(m))
        if shuffling:
            np.random.shuffle(order)
        return order

    def batch(self, X, y, batch_size, index, order):
        """
        Get a batch of data.
        args:
            X: input to the model
            y: labels
            batch_size: batch size
            index: index of the batch
                e.g: if batch_size = 3 and index = 1 then the batch will be from index [3, 4, 5]
            order: order of the data
        returns:
            bx, by: batch of data
        """
        # DONE: Implement batch
        # hint last index of the batch check for the last batch
        last_index = min(index + batch_size, len(order))
        batch = order[index:last_index]
        # NOTICE: inputs are 4 dimensional or 2 demensional        
        if len(X.shape) == 4:
            bx = X[batch,:,:,:]
        else:
            bx = X[batch,:]
        by = y[batch,:]
        return bx, by

    def compute_loss(self, X, y, batch_size):
        """
        Compute the loss.
        args:
            X: input to the model
            y: labels
            Batch_Size: batch size
        returns:
            loss
        """
        # DONE: Implement compute loss
        m = X.shape[0]
        order = self.shuffle(m, False)
        cost = 0
        for b in range(0, m, batch_size):
            bx, by = self.batch(X, y, batch_size, b, order)
            tmp = self.forward(bx)
            AL = tmp[-1]
            cost += self.criterion(AL, by)
        return cost / (m // batch_size)

    def train(self, X, y, epochs, val=None, batch_size=32, shuffling=False, verbose=1, save_after=None):
        """
        Train the model.
        args:
            X: input to the model
            y: labels
            epochs: number of epochs
            val: validation data
            batch_size: batch size
            shuffling: if True shuffle the data
            verbose: if 1 print the loss after each epoch
            save_after: save the model after training
        """
        # DONE: Implement training
        train_cost = []
        val_cost = []
        # NOTICE: if your inputs are 4 dimensional m = X.shape[0] else m = X.shape[1]
        m = X.shape[0]
        for e in tqdm.tqdm(range(epochs)):
            order = self.shuffle(m, shuffling)
            cost = self.one_epoch(X, y, batch_size)
            train_cost.append(cost)
            if val is not None:
                val_cost.append(self.compute_loss(val[0], val[1], batch_size))
            if verbose == 1:
                print("Epoch {}: train cost = {}".format(e+1, cost))
                if val is not None:
                    print("Epoch {}: val cost = {}".format(e+1, val_cost[-1]))
            elif verbose > 1 and e % verbose == 0:
                print("Epoch {}: train cost = {}".format(e+1, cost))
                if val is not None:
                    print("Epoch {}: val cost = {}".format(e+1, val_cost[-1]))
        
        if save_after is not None:
            self.save(save_after)
        return train_cost, val_cost
    
    def predict(self, X):
        """
        Predict the output of the model.
        args:
            X: input to the model
        returns:
            predictions
        """
        # DONE: Implement prediction
        tmp = self.forward(X)
        return np.argmax(tmp[-1], axis=-1)

