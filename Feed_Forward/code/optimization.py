"""All the optimization methods go here.

"""

from __future__ import division, print_function, absolute_import
import random
import numpy as np


class SGD(object):
    """Mini-batch stochastic gradient descent.

    Attributes:
        learning_rate(float): the learning rate to use.
        batch_size(int): the number of samples in a mini-batch.

    """

    def __init__(self, learning_rate, batch_size):
        self.learning_rate = float(learning_rate)
        self.batch_size = batch_size

    def __has_parameters(self, layer):
        return hasattr(layer, "W")

    def compute_gradient(self, x, y, graph, loss):
        """ Compute the gradients of network parameters (weights and biases)
        using backpropagation.

        Args:
            x(np.array): the input to the network.
            y(np.array): the ground truth of the input.
            graph(obj): the network structure.
            loss(obj): the loss function for the network.

        Returns:
            dv_Ws(list): a list of gradients of the weights.
            dv_bs(list): a list of gradients of the biases.

        """

        # TODO: Backpropagation code
        alist = []
        dws = []
        dbs = []
        a = np.array(x)
        alist.append(a)
        ls = [layer for layer in graph]
        a = ls[0].forward(a)
        alist.append(a)
        a = ls[1].forward(a)
        alist.append(a)
        a = ls[2].forward(a)
        alist.append(a)
        a = ls[3].forward(a)
        alist.append(a)
        da = loss.backward(a,y)
        dy = ls[3].backward(alist[3], da)
        dx, dw, db = ls[2].backward(alist[2],dy)
        dws.append(dw)
        dbs.append(db)
        dy = ls[1].backward(alist[1],dx)
        dx, dw, db = ls[0].backward(alist[0],dy)
        dws.append(dw)
        dbs.append(db)
        dws.reverse()
        dbs.reverse()
        return dws, dbs
#        dws = [0,0]
#        dbs = [0,0]
#        for xi,yi in x,y:
#            dw,db = get_para(graph ,xi, yi)
#            dws = dws+dw
#            dbs = dbs+db
#        dws = dws/self.batch_size
#        dbs = dbs/self.batch_size
#        return dws, dbs
        #pass

    def optimize(self, graph, loss, training_data):
        """ Perform SGD on the network defined by 'graph' using
        'training_data'.

        Args:
            graph(obj): a 'Graph' object that defines the structure of a
                neural network.
            loss(obj): the loss function for the network.
            training_data(list): a list of tuples ``(x, y)`` representing the
                training inputs and the desired outputs.

        """

        # Network parameters
        Ws = [layer.W for layer in graph if self.__has_parameters(layer)]
        bs = [layer.b for layer in graph if self.__has_parameters(layer)]

        # Shuffle the data to make sure samples in each batch are not
        # correlated
        random.shuffle(training_data)
        n = len(training_data)

        batches = [
            training_data[k:k + self.batch_size]
            for k in xrange(0, n, self.batch_size)
        ]
        # TODO: SGD code
        for batch in batches:
            for b in batch:
                wgra,bgra = self.compute_gradient(b[0], b[1], graph, loss)
                graph.layers[0].W = graph.layers[0].W-self.learning_rate*wgra[0]
                graph.layers[0].b = graph.layers[0].b-self.learning_rate*bgra[0]
                graph.layers[2].W = graph.layers[2].W-self.learning_rate*wgra[1]
                graph.layers[2].b = graph.layers[2].b-self.learning_rate*bgra[1]
        #pass
