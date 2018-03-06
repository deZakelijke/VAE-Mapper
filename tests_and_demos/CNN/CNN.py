import numpy as np
import math

class CNN:
    '''
    A Convolutional neural network class
    
    Made by Micha de Groot
    '''

    def __init__(self, layer_shapes = [(3,1)]):
        self.in_size = layer_shapes[0][0]
        self.out_size = layer_shapes[-1][-1]
        self.weights = self.__init_weights__(layer_shapes)


    def feed_forward(self, x):
        '''
            Perform a feed forward pass on a data entry
        '''
        if not np.shape(x)[0] is self.in_size:
            raise IndexError("Inpus has mismatching shape for feed forward")
        for weight in self.weights:
            result = self.__activation__(x, weight)
        return result


    def __init_weights__(self, layer_shapes):
        '''
            Initialize the weights of the network
        '''
        for i in range(len(layer_shapes) - 1):
            if not layer_shapes[i][1] is layer_shapes[i + 1][0]:
                raise ValueError("Mismatching shapes of network")
        theta_list = []
        for layer in layer_shapes:
            theta_list.append(np.random.rand(*layer))
        return theta_list


    def __activation__(self, X, Theta):
        '''
            Calculate the activation values of a node.
        '''
        return 1.0 / (1.0 + np.exp( -np.dot(X, Theta)))


if __name__ == '__main__':
    layers = [(3, 3), (3, 1)]
    network = CNN(layers)
    result = network.feed_forward([1, 2, -1])
    print(result)

