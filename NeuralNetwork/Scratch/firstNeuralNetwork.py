import numpy as np
import matplotlib.pyplot as plt
from loss_functions import *

class Layer:
     def __init__(self, matrix, activation, activationD, numNodes, index):
        """
        Constructor for Neural Network Layer

        This function initializes a Layer with a matrix of biases and weights to represent nodes

        :param biases: values of node biases matrix
        :param weights: values of node weight matrix
        :param activation: function of node activation function
        :param activationD: derivative of node activation function
        :param numNodes: number of nodes in the layer
        :param index: index of layer in network
        :param value: value of ouputs of this layer
        """
        
        self.matrix = matrix
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
        """

        #Create hidden layer with random biases, random weights, given size (plus bias unit) and activation function, and assign index
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
        """



        #Initialize weights and biases using He initialization method; Biases set with zero values

        matrix = np.zeros((numNodes, self.Network[-1].numNodes))

        for node in range(numNodes):
            matrix[node, :-1] = he_initializer(self.Network[-1].numNodes - 1)

        #Create hidden layer with random biases, random weights, given size and activation function, and assign index
        hiddenLayer = Layer(matrix, activation, activationD, numNodes, len(self.Network))
       
        #Add to network
        self.Network.append(hiddenLayer)

    def addOutput(self, lenOutput, activation, activationD):
        """
        Add Ouput Layer to Neural Network

        This function adds a hidden layer to the neural network.

        :param lenOutput: Number of Output Neurons
        :param activation: Function of Node Activation function
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
            layer.values = layer.matrix @ prev_layer.values
            if layer.activation:
                layer.values = layer.activation(layer.values)
            
        #Return the output layer values
        return self.Network[-1].values

    def backProp(self, predicted, real):
        """
        Run 1 backward pass through network

        This function uses the chain rule and gradient descent to iteratively update the network biases and weights using the final error

        :param predicted: Value(s) output by forward pass of model
        :param real: Actual value(s) of function
        """

        # Compute the error
        error = MSE(predicted, real)

        gradient_matrix = np.ones(self.Network[-1].matrix.shape) * -2 * error

        if self.Network[-1].activationD is not None:
            gradient_matrix *= self.Network[-1].activationD(self.Network[-1].values)

        # Backpropagate the error through the layers
        for i in range(len(self.Network) - 2, 0, -1):
            current_layer = self.Network[i]
            prev_layer = self.Network[i - 1]
            # Compute the gradients for weights and biases

            if i == len(self.Network) - 1:
                if current_layer.activationD is not None:
                    activation_derivative = current_layer.activationD(current_layer.values)

                    gradient = -2 * error * activation_derivative

                else:
                    gradient = -2 * error
            elif i != 0:
                # Compute weight and bias gradients
                weight_gradient = np.outer(gradient, prev_layer.values).T
                bias_gradient = gradient

                # Update weights and biases using gradients and learning rate

                current_layer.weights -= self.learningRate * weight_gradient
                current_layer.biases -= self.learningRate * bias_gradient
                if i > 1:
                    error = np.dot(current_layer.weights, gradient)[0]
                
def testFunction(x):
    """
        Define a function to model in main function

        This function is used to model the function that the neural network will try to learn

        :param value: input value to function
    """
    return np.sin(x)

def main():

    num_vals = 1
    num_epochs = 100
    #Define the input and output values
    x = [np.array(i) for i in np.linspace(-5, 5, num=num_vals)]

    #Add bias to input
    y = testFunction(x)

    NN = NeuralNetwork(1, 0.1)
    NN.addHidden(3, ReLU, ReLUD)
    NN.addOutput(1, ReLU, ReLUD)

    for epoch in range(num_epochs):
        for i in range(num_vals):
            NN.setInput(x[i])
            NN.setInputTest(y[i])

            prediction = NN.feedForward()[0]
            print(prediction)
            true_val = y[i]
            NN.backProp(prediction, true_val)

    x_test = [[i] for i in np.linspace(-5, 5, num=num_vals)]
    y_test = testFunction(x_test)
    
    test = []
    for i in range(num_vals):
        NN.setInput(x_test[i])
        prediction = NN.feedForward()[0]
        test.append(prediction)

    #plt.plot(x_test, y_test)
    #plt.plot(x_test, test)
    #plt.show()
if __name__ == "__main__":
    main()