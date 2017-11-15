"""All the layer functions go here.
"""

from __future__ import division, print_function, absolute_import
import numpy as np


class FullyConnected(object):
    """Fully connected layer 'y = Wx + b'.

    Arguments:
        shape(tuple): the shape of the fully connected layer. shape[0] is the
            output size and shape[1] is the input size.

    Attributes:
        W(np.array): the weights of the fully connected layer. An n-by-m matrix
            where m is the input size and n is the output size.
        b(np.array): the biases of the fully connected layer. A n-by-1 vector
            where n is the output size.

    """

    def __init__(self, shape):
        # Gaussian initilization
        self.W = np.random.randn(*shape)
        self.b = np.random.randn(shape[0], 1)

        #Pitfall
        #self.W = np.zeros_like(self.W)
        #self.b = np.zeros_like(self.b)

        #small number
        #self.W = 0.01*np.random.randn(*shape)
        #self.b = 0.01*np.random.randn(shape[0], 1)

        #Calibrating the variances with 1/sqrt(n)
        #n = shape[0]*shape[1]
        #self.W = np.random.randn(*shape)/(n**0.5)
        #self.b = np.random.randn(shape[0], 1)/(n**0.5)

    def forward(self, x):
        """Compute the layer output.

        Args:
            x(np.array): the input of the layer.

        Returns:
            The output of the layer.

        """

        # TODO: Forward code
        Y = np.dot(self.W,x)
        Y = np.add(Y, self.b)
        return Y
        #pass

    def backward(self, x, dv_y):
        """Compute the gradients of weights and biases and the gradient with
        respect to the input.

        Args:
            x(np.array): the input of the layer.
            dv_y(np.array): The derivative of the loss with respect to the
                output.

        Returns:
            dv_x(np.array): The derivative of the loss with respect to the
                input.
            dv_W(np.array): The derivative of the loss with respect to the
                weights.
            dv_b(np.array): The derivative of the loss with respect to the
                biases.

        """

        # TODO: Backward code
        dv_x = np.dot(self.W.T,dv_y);
        dv_W = np.dot(dv_y,x.T)
        dv_b = dv_y;

        return dv_x, dv_W, dv_b

        #pass


class Sigmoid(object):
    """Sigmoid function 'y = 1 / (1 + exp(-x))'

    """

    def forward(self, x):
        """Compute the layer output.

        Args:
            x(np.array): the input of the layer.

        Returns:
            The output of the layer.

        """

        # TODO: Forward code
        Y = np.divide(1,np.add(1,np.exp(np.negative(x))))
        return Y
        #pass

    def backward(self, x, dv_y):
        """Compute the gradient with respect to the input.

        Args:
            x(np.array): the input of the layer.
            dv_y(np.array): The derivative of the loss with respect to the
                output.

        Returns:
            The derivative of the loss with respect to the input.

        """

        # TODO: Backward code
        k=np.divide(1, np.add(1, np.exp(np.negative(x))))
        sk = np.multiply(k,np.subtract(1,k))
        return np.multiply(dv_y ,sk)

        #pass
