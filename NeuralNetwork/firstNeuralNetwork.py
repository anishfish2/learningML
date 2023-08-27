import numpy as np

class Layer:
     def __init__(self, bias, weights, activation, index):
        """
        Constructor for Neural Network Layer

        This function initializes a Layer with a matrix of biases and weights to represent nodes

        :param bias: values of node biases matrix
        :param weights: values of node weight matrix
        :param activation: function of node activation function
        :param index: index of layer in network
        """
        
        self.bias = bias
        self.weights = weights
        self.activation = activation
        self.index = index
        

class NeuralNetwork:
    def __init__(self, lenInput):
        """
        Constructor for Neural Network

        This function initializes the neural network with an input layer.

        :param lenInput: Number of Input Neurons
        """
        
        #Create the input layer with no biases, no weights, no activation function, and index 0
        inputLayer = Layer(np.zeros((1, lenInput)), np.identity((lenInput)), None, 0)

        #Create the Network
        self.Network = [inputLayer]



    def addHidden(self, numNodes, activation):
        """
        Add Hidden Layer to Neural Network

        This function adds a hidden layer to the neural network.

        :param numNodes: Number of Hidden Layer Neurons
        :param numNodes: Number of Hidden Layer Neurons
        :param activation: Function of Node Activation function
        """
        
        #Create hidden layer with random biases, random weights, given size and activation function, and assign index
        inputLayer = Layer(np.zeros((1, numNodes)), np.identity((numNodes)), None, 0)
       

    def addOutput(self, lenOutput, activation):
        """
        Add Ouput Layer to Neural Network

        This function adds a hidden layer to the neural network.

        :param lenOutput: Number of Output Neurons
        :param activation: Function of Node Activation function
        """
        
      

    def feedForward(self, inputs):
        """
        Run 1 forward pass through network

        This function takes values, runs them through current state of network and returns output values

        :param inputs: Input values to Input Layer
        """







def ReLU(x):
    """
        Standard implementation of ReLU

        This function returns the max of x and 0
        
        :param x: Value to apply ReLU
    """
    return max(0, x)

def sigmoid(x):
    """
        Standard implementation of Sigmoid

        This function maps values between 0 and 1
        
        :param x: Value to apply Sigmoid
    """
    return 1 / (1 + np.exp(-x))

def main():
