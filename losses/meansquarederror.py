import numpy as np

class MeanSquaredError:
    def __init__(self):
        pass

    def compute(self, y_pred, y_true):
        """
        computes the mean squared error loss
            args:
                y_pred: predicted labels (n_classes, batch_size)
                y_true: true labels (n_classes, batch_size)
            returns:
                mean squared error loss
        """
        # Get the batch size
        batch_size = y_pred.shape[1]
        
        # Compute the mean squared error loss
        cost = (1/(2*batch_size)) * np.sum((y_pred - y_true) ** 2)
        
        return np.squeeze(cost)
    
    def backward(self, y_pred, y_true):
        """
        computes the derivative of the mean squared error loss
            args:
                y_pred: predicted labels (n_classes, batch_size)
                y_true: true labels (n_classes, batch_size)
            returns:
                derivative of the mean squared error loss
        """
        # Compute the derivative of the mean squared error loss
        db = (1/y_pred.shape[1]) * (y_pred - y_true)
        
        return db
