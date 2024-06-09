import numpy as np
import matplotlib.pyplot as plt
from loss_functions import *

class Layer:
     def __init__(self, matrix, activation, activationD, numNodes, index):
        """
        Constructor for Neural Network Layer

        This function initializes a Layer with a matrix of biases and weights to represent nodes

        :param weights: values of node weight matrix
        :param activation: function of node activation function
        :param activationD: derivative of node activation function
        :param numNodes: number of nodes in the layer
        :param index: index of layer in network
        :param value: value of ouputs of this layer
        """
        
        self.weight_matrix = matrix
        self.weight_gradient_matrix = None
        self.value_gradient_matrix = None
        self.activation = activation
        self.activationD = activationD
        self.numNodes = numNodes
        self.index = index
        self.values = None
        
class NeuralNetwork:
    def __init__(self, lenInput, learningRate):
        """
        Constructor for Neural Network

        This function initializes the neural network with an input layer.

        :param lenInput: Number of Input Layer Neurons
        :param learningRate: Learning Rate of Neural Network
        """

        #Create input layer with no biases, no weights, given size (plus bias unit) and activation function, and assign index
        inputLayer = Layer(None, None, None, lenInput + 1, 0)

        #Create the Network
        self.Network = [inputLayer]
        self.learningRate = learningRate

    def setInput(self, input):
        """
        Add 1 training example to model

        This function adds one input to feed forward and train network on

        :param inputs: 1 input
        """

        self.input_values = input
    
    def setInputTest(self, input_test):
        """
        Add 1 test example to model

        This function adds one set of tests to train inputs to train network on

        :param input_test: 1 set of input tests
        """

        self.input_values_test = input_test

    def addHidden(self, numNodes, activation, activationD):
        """
        Add Hidden Layer to Neural Network

        This function adds a hidden layer to the neural network.

        Using He Initialization method: '-sqrt(2 / fan_in) and sqrt(2 / fan_in)'

        :param numNodes: Number of Hidden Layer Neurons
        :param activation: Function of Node Activation function
        :param activationD: Derivative of Node Activation function
        """

        #Initialize weights and biases using He initialization method; Biases set with zero values

        matrix = np.zeros((numNodes, self.Network[-1].numNodes))

        for node in range(numNodes):
            matrix[node, :-1] = he_initializer(self.Network[-1].numNodes - 1)

        #Create hidden layer with random biases, random weights, given size and activation function, and assign index
        hiddenLayer = Layer(matrix, activation, activationD, numNodes + 1, len(self.Network))
       
        #Add to network
        self.Network.append(hiddenLayer)

    def addOutput(self, lenOutput, activation, activationD):
        """
        Add Ouput Layer to Neural Network

        This function adds a hidden layer to the neural network.

        :param lenOutput: Number of Output Neurons
        :param activation: Function of Node Activation function
        :param activationD: Derivative of Node Activation function
        """

        #Initialize weights and biases using He initialization method; Biases set with zero values
        matrix = np.zeros((lenOutput, self.Network[-1].numNodes))

        for node in range(lenOutput):
            matrix[node, :-1] = he_initializer(self.Network[-1].numNodes - 1)

        #Create hidden layer with random biases, random weights, given size and activation function, and assign index
        outputLayer = Layer(matrix, activation, activationD, lenOutput, len(self.Network))
       
        #Add to network
        self.Network.append(outputLayer)
        
    def feedForward(self):
        """
        Run 1 forward pass through network

        This function takes values, runs them through current state of network and returns output values

        :param inputs: Input values to Input Layer
        """
        self.Network[0].values = np.append(np.array([self.input_values]), 1)
        for i in range(1, len(self.Network)):
            layer = self.Network[i]
            prev_layer = self.Network[i-1]
            layer.values = prev_layer.values @ layer.weight_matrix.T 
            
            if layer.activation:
                layer.values = layer.activation(layer.values)
            
            # Add bias unit
            if i != len(self.Network) - 1:
                layer.values = np.append(layer.values, 1)

        #Return the output layer values
        return self.Network[-1].values, 1

    def backProp(self, actual, predicted, lossFunction, lossFunctionD, lr):
        """
        Run 1 backward pass through network 

        This function uses the chain rule and gradient descent to iteratively update the network biases and weights using the final error

        :param predicted: Value(s) output by forward pass of model
        :param actual: Actual value(s) of function
        """

        # Compute the error
        loss = lossFunction(actual, predicted)

        # Compute the gradient of the error with respect to the output layer
        for index in range(len(self.Network) - 1, 0, -1):
            layer = self.Network[index] 
            prev_layer = self.Network[index - 1]

            if index == len(self.Network) - 1:
                layer.value_gradient_matrix = layer.activationD(layer.values) * lossFunctionD(actual, predicted)
                layer.weight_gradient_matrix = np.reshape(layer.value_gradient_matrix, (-1, 1)) @ np.reshape(prev_layer.values, (1, -1))
                layer.weight_matrix = layer.weight_matrix - layer.value_gradient_matrix * lr 
            else:
                print("Calculating value gradient matrix", layer.weight_matrix.shape, self.Network[index + 1].value_gradient_matrix.shape, self.Network[index + 1].weight_matrix[:, :-1].shape, prev_layer.values.shape)
                layer.value_gradient_matrix = layer.activationD(layer.values[:-1]) * (np.reshape(self.Network[index + 1].value_gradient_matrix, (-1,1)) @ self.Network[index + 1].weight_matrix[:, :-1]).squeeze()
                print("Layer value gradient matrix", layer.value_gradient_matrix.shape, layer.weight_matrix.shape, prev_layer.values.shape)
                layer.weight_gradient_matrix = np.reshape(layer.value_gradient_matrix, (-1, 1)) @ np.reshape(prev_layer.values[:-1], (1, -1))
                print(layer.weight_matrix.shape, layer.value_gradient_matrix.shape, prev_layer.values.shape)
                layer.weight_matrix = layer.weight_matrix - layer.value_gradient_matrix * lr
        
def testFunction(x):
    """
        Define a function to model in main function

        This function is used to model the function that the neural network will try to learn

        :param value: input value to function
    """
    return np.sin(x)

def main():

    num_vals = 1
    num_epochs = 3

    #Define the input and output values
    x = [np.array([i]) for i in np.linspace(-5, 5, num=num_vals)]

    y = testFunction(x)

    NN = NeuralNetwork(1, 0.1)
    NN.addHidden(3, ReLU, ReLUD)
    NN.addOutput(1, ReLU, ReLUD)

    for epoch in range(num_epochs):
        
        for i in range(num_vals):
            print("Epoch: ", epoch + 1, "Num Val:", i + 1)
            NN.setInput(x[i])
            NN.setInputTest(y[i])

            prediction = NN.feedForward()[0]
            true_val = y[i]
            print(prediction, true_val)
            NN.backProp(true_val, prediction, MSE, MSED, lr=0.01)

    # x_test = [[i] for i in np.linspace(-5, 5, num=num_vals)]
    # y_test = testFunction(x_test)
    
    # test = []
    # for i in range(num_vals):
    #     NN.setInput(x_test[i])
    #     prediction = NN.feedForward()[0]
    #     test.append(prediction)

    #plt.plot(x_test, y_test)
    #plt.plot(x_test, test)
    #plt.show()
if __name__ == "__main__":
    main()