import numpy as np

class MaxPool2D:
    def __init__(self, kernel_size=(3, 3), stride=(1, 1), mode="max"):
        """
        Max pooling layer.
            args:
                kernel_size: size of the kernel
                stride: stride of the kernel
                mode: max or average
            Question:Why we don't need to set name for the layer?
            Answer:The max pooling operation is performed using a pre-implemented layer,
                typically called "MaxPooling2D" or something similar. so it's not necessary to
                name the layer, and it does not have any efect on model correct functionality.
        """
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.mode = mode
    
    def target_shape(self, input_shape):
        """
        Calculate the shape of the output of the layer.
            args:
                input_shape: shape of the input
            returns:
                output_shape: shape of the output
        """
        # DONE: Implement target_shape
        (batch_size, H_prev, W_prev, C_prev) = input_shape
        (f_h, f_w) = self.kernel_size
        strideh, stridew = self.stride
        H = int(1 + (H_prev - f_h) / strideh)
        W = int(1 + (W_prev - f_w) / stridew)
        return (batch_size, H, W, C_prev)
    
    def forward(self, A_prev):
        """
        Forward pass for max pooling layer.
            args:
                A_prev: activations from previous layer (or input data)
            returns:
                A: output of the max pooling layer
        """
        # Implement forward pass for max pooling layer
        (batch_size, H_prev, W_prev, C_prev) = A_prev.shape
        (kernel_size_h, kernel_size_w) = self.kernel_size
        strideh, stridew = self.stride
        H = int(1 + (H_prev - kernel_size_h) / strideh)
        W = int(1 + (W_prev - kernel_size_w) / stridew)
        A = np.zeros((batch_size, H, W, C_prev))
        for i in range(batch_size):
            for h in range(H):
                h_start = h * strideh
                h_end = h_start + kernel_size_h
                for w in range(W):
                    w_start = w * stridew
                    w_end = w_start + kernel_size_w
                    for c in range(C_prev):
                        a_prev_slice = A_prev[i, h_start:h_end, w_start:w_end, c]
                        if self.mode == "max":
                            A[i, h, w, c] = np.max(a_prev_slice)
                        elif self.mode == "average":
                            A[i, h, w, c] = np.mean(a_prev_slice)
                        else:
                            raise ValueError("Invalid mode")

        return A
    
    def create_mask_from_window(self, x):
        """
        Create a mask from an input matrix x, to identify the max entry of x.
            args:
                x: numpy array
            returns:
                mask: numpy array of the same shape as window, contains a True at the position corresponding to the max entry of x.
        """
        # DONE:Implement create_mask_from_window
        mask = x == np.max(x)
        return mask
    
    def distribute_value(self, dz, shape):
        """
        Distribute the input value in the matrix of dimension shape.
            args:
                dz: input scalar
                shape: the shape (n_H, n_W) of the output matrix for which we want to distribute the value of dz
            returns:
                a: distributed value
        """
        # DONE:Implement distribute_value
        (n_H, n_W) = shape
        average = dz / (n_H * n_W)
        a = np.ones(shape) * average
        return a
    
    def backward(self, dA, A_prev):
        """
        Backward pass for max pooling layer.
            args:
                dA: gradient of cost with respect to the output of the max pooling layer
                A_prev: activations from previous layer (or input data)
            returns:
                dA_prev: gradient of cost with respect to the input of the max pooling layer
        """
        # DONE: Implement backward pass for max pooling layer
        (batch_size, H, W, C) = dA.shape
        (f_h, f_w) = self.kernel_size
        strideh, stridew = self.stride
        batch_size, H_prev, W_prev, C_prev = A_prev.shape
        dA_prev = np.zeros((batch_size, H_prev, W_prev, C_prev))
        for i in range(batch_size):
            for h in range(H):
                for w in range(W):
                    for c in range(C):
                        h_start = h
                        h_end = h_start + f_h
                    w_start = w * stridew
                    w_end = w_start + f_w
                    
                    if self.mode == "max":
                        a_prev_slice = A_prev[i, h_start:h_end, w_start:w_end, c]
                        mask = self.create_mask_from_window(a_prev_slice)
                        dA_prev[i, h_start:h_end, w_start:w_end, c] += np.multiply(mask, dA[i, h, w, c])
                        
                    elif self.mode == "average":
                        shape = (f_h, f_w)
                        dz = dA[i, h, w, c]
                        dA_prev[i, h_start:h_end, w_start:w_end, c] += self.distribute_value(dz, shape)
                        
                    else:
                        raise ValueError("Invalid mode")

        return dA_prev, None
    
    
