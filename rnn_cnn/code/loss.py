"""All the loss functions go here.

"""

from __future__ import division, print_function, absolute_import

import numpy as np


class LogSoftmax(object):
    """The log softmax (sparse) loss 'L = -log(softmax(y_gt))'.

    """
    def __init__(self, name="LogSoftmax"):
        self.name = name

    def forward(self, y, gt):
        """Compute the log softmax loss.

        Args:
            y (np.array): the output from previous layer. It is a column
                vector.
            gt (int): the ground truth. Note it is an integer indicate the
                label of an input data.

        Return:
            The log softmax loss.

        """

        # TODO: Put you code below
        #pass
        p = np.divide(np.exp(y),np.sum(np.exp(y)))
        l = np.negative(np.log(p[gt, 0]))

        return l

    def backward(self, y, gt):
        """Compute the derivative of the log softmax loss.

        Args:
            y (np.array): the output from previous layer. It is a column
                vector.
            gt (int): the ground truth. Note it is an integer indicate the
                label of an input data.

        Returns:
            The derivative of the loss with respect to the y.

        """

        # TODO: Put you code below
        #pass
        dv_y = np.divide(np.exp(y), np.sum(np.exp(y)))

        dv_y[gt] = dv_y[gt] - 1

        return dv_y

