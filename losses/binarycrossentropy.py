import numpy as np

class BinaryCrossEntropy:
    def __init__(self) -> None:
        pass

    def compute(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        """
        Computes the binary cross entropy loss.
            args:
                y: true labels (n_classes, batch_size)
                y_hat: predicted labels (n_classes, batch_size)
            returns:
                binary cross entropy loss
        """
        # Get the batch size
        batch_size = y.shape[1]
        
        # Compute the binary cross entropy loss
        cost = -(1/batch_size) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        
        return np.squeeze(cost)

    def backward(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the binary cross entropy loss.
            args:
                y: true labels (n_classes, batch_size)
                y_hat: predicted labels (n_classes, batch_size)
            returns:
                derivative of the binary cross entropy loss
        """
        # Compute the derivative of the binary cross entropy loss
        db = np.divide(y_hat - y, y_hat * (1 - y_hat))
        
        return db


