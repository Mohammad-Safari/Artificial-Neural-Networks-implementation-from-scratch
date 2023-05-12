import numpy as np
from abc import ABC, abstractmethod

class Activation(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def forward(self, Z: np.ndarray) -> np.ndarray:
        """
        Forward pass for activation function.
            args:
                Z: input to the activation function
            returns:
                A: output of the activation function
        """
        pass

    @abstractmethod
    def backward(self, dA: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """
        Backward pass for activation function.
            args:
                dA: derivative of the cost with respect to the activation
                Z: input to the activation function
            returns:
                derivative of the cost with respect to Z
        """
        pass

class Sigmoid(Activation):
    def forward(self, Z: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function.
            args:
            x: input to the activation function
            returns:
                sigmoid(x)
        """
        A = 1 / (1 + np.exp(-Z))
        return A

    def backward(self, dA: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """
        Backward pass for sigmoid activation function.
            args:
                dA: derivative of the cost with respect to the activation
                Z: input to the activation function
            returns:
                derivative of the cost with respect to Z
        """
        A = self.forward(Z)
        dZ = dA * A * (1 - A)
        return dZ
    

class ReLU(Activation):
    def forward(self, Z: np.ndarray) -> np.ndarray:
        """
        ReLU activation function.
            args:
                x: input to the activation function
            returns:
                relu(x)
        """
        A = np.maximum(0, Z)
        return A

    def backward(self, dA: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """
        Backward pass for ReLU activation function.
            args:
                dA: derivative of the cost with respect to the activation
                Z: input to the activation function
            returns:
                derivative of the cost with respect to Z
        """
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ
    
    

class Tanh(Activation):
    def forward(self, Z: np.ndarray) -> np.ndarray:
        """
        Tanh activation function.
            args:
                x: input to the activation function
            returns:
                tanh(x)
        """
        A = np.tanh(Z)
        return A

    def backward(self, dA: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """
        Backward pass for tanh activation function.
            args:
                dA: derivative of the cost with respect to the activation
                Z: input to the activation function
            returns:
                derivative of the cost with respect to Z
        """
        A = self.forward(Z)
        dZ = (1 - A**2) * dA
        return dZ
    
class LinearActivation(Activation):
    def forward(self, Z: np.ndarray) -> np.ndarray:
        """
        Linear activation function.
            args:
                x: input to the activation function
            returns:
                x
        """
        A = Z
        return A

    def backward(self, dA: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """
        Backward pass for linear activation function.
            args:
                dA: derivative of the cost with respect to the activation
                Z: input to the activation function
            returns:
                derivative of the cost with respect to Z
        """
        dZ = np.array(dA, copy=True)
        return dZ

def get_activation(activation: str) -> tuple:
    """
    Returns the activation function and its derivative.
        args:
            activation: activation function name
        returns:
            activation function and its derivative
    """
    if activation == 'sigmoid':
        return Sigmoid(), Sigmoid().backward
    elif activation == 'relu':
        return ReLU(), ReLU().backward
    elif activation == 'tanh':
        return Tanh(), Tanh().backward
    elif activation == 'linear':
        return LinearActivation(), LinearActivation().backward
    else:
        raise ValueError('Activation function not supported')
