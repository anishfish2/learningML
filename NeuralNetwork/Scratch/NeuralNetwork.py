import numpy as np
import matplotlib.pyplot as plt
from loss_functions import *
from tqdm import tqdm

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
    def __init__(self, lenInput):
        """
        Constructor for Neural Network

        This function initializes the neural network with an input layer.

        :param lenInput: Number of Input Layer Neurons
        """

        inputLayer = Layer(None, None, None, lenInput + 1, 0)

        self.Network = [inputLayer]

    def addLayer(self, numNodes, activation, activationD, isOutput = False):
        """
        Add Layer to Neural Network

        This function adds a layer to the neural network.

        Using He Initialization method: '-sqrt(2 / fan_in) and sqrt(2 / fan_in)'

        :param numNodes: Number of Layer Neurons
        :param activation: Function of Node Activation function
        :param activationD: Derivative of Node Activation function
        """

        #Initialize weights and biases using He initialization method; Biases set with zero values
        #Create hidden layer with random biases, random weights, given size and activation function, and assign index
        if not isOutput:
            matrix = np.zeros((numNodes + 1, self.Network[-1].numNodes))
            for node in range(numNodes):
                matrix[node, :-1] = he_initializer(self.Network[-1].numNodes - 1)

            layer = Layer(matrix, activation, activationD, numNodes + 1, len(self.Network))
        else:
            matrix = np.zeros((numNodes, self.Network[-1].numNodes))
            for node in range(numNodes):
                matrix[node, :] = he_initializer(self.Network[-1].numNodes)
            layer = Layer(matrix, activation, activationD, numNodes, len(self.Network))

            
        #Add to network
        self.Network.append(layer)
        
    def feedForward(self, input):
        """
        Run 1 forward pass through network

        This function takes values, runs them through current state of network and returns output values

        :param inputs: Input values to Input Layer
        """
        self.Network[0].values = np.append(np.array([input]), 1)
        for i in range(1, len(self.Network)):
            layer = self.Network[i]
            prev_layer = self.Network[i-1]
            layer.values = prev_layer.values @ layer.weight_matrix.T 

            if layer.activation:
                layer.values = layer.activation(layer.values)
            
            # Add bias unit
            # if i != len(self.Network) - 1:
            #     layer.values = np.append(layer.values, 1)

        #Return the output layer values
        return self.Network[-1].values

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
            next_layer = self.Network[index - 1]
            if index == len(self.Network) - 1:
                if layer.activation:
                    layer.value_gradient_matrix = layer.activationD(layer.values) * lossFunctionD(actual, predicted)
                else:
                    layer.value_gradient_matrix = lossFunctionD(actual, predicted)
                layer.weight_gradient_matrix = np.reshape(layer.value_gradient_matrix, (-1, 1)) @ np.reshape(next_layer.values, (1, -1))
                layer.weight_matrix = layer.weight_matrix - layer.value_gradient_matrix * lr 
            else:
                # if index == len(self.Network) - 2:
                #     if layer.activation:
                #         layer.value_gradient_matrix = layer.activationD(layer.values) * (np.reshape(self.Network[index + 1].value_gradient_matrix, (1,-1)) @ self.Network[index + 1].weight_matrix)
                #     else:
                #         layer.value_gradient_matrix = np.reshape(self.Network[index + 1].value_gradient_matrix, (-1,1)) @ self.Network[index + 1].weight_matrix
                # else:
                if layer.activation:
                    layer.value_gradient_matrix = layer.activationD(layer.values) * (np.reshape(self.Network[index + 1].value_gradient_matrix, (1,-1)) @ self.Network[index].weight_matrix.T)
                else:
                    layer.value_gradient_matrix = np.reshape(self.Network[index + 1].value_gradient_matrix[:, :-1], (1,-1)) @ self.Network[index + 1].weight_matrix
                layer.weight_gradient_matrix = np.reshape(layer.value_gradient_matrix[:, :-1], (-1, 1)) @ np.reshape(next_layer.values, (1, -1))

                layer.weight_matrix = layer.weight_matrix - layer.weight_gradient_matrix * lr

        return loss

def testFunction(x):
    """
        Define a function to model in main function

        This function is used to model the function that the neural network will try to learn

        :param value: input value to function
    """
    return np.sin(x)

def main():

    num_vals = 10000
    num_epochs = 1

    #Define the input and output values
    x = np.array([np.array([i]) for i in np.linspace(-5, 5, num=num_vals)])
    y = testFunction(x)

    NN = NeuralNetwork(1)
    NN.addLayer(8, tanh, tanhD)
    NN.addLayer(1, identity, identityD, True)

    test = []
    train = []
    epoch_loss = []
    
    for epoch in tqdm(range(num_epochs)):
        total_loss = 0
       
        for i in range(num_vals):

            prediction = NN.feedForward(x[i])
            true_val = y[i]
            loss = NN.backProp(true_val, prediction, MSE, MSED, lr=0.001)

            # if i % 1000 == 0:
            #     print("Epoch: ", epoch + 1, "loss", loss, "Prediction: ", prediction, "True Value: ", true_val)
            total_loss += loss

        x_test = np.array([[i] for i in np.linspace(-5, 5, num=num_vals)])
        y_test = testFunction(x_test)
    
        y_preds = []
        for i in range(num_vals):
            y_preds.append(NN.feedForward(x_test[i])[0])

        epoch_loss.append(total_loss)

    
        
    # for index, epoch in enumerate(train):
    #     plt.plot(x, y)
    #     plt.plot(x, epoch, label="Epoch: " + str(train.index(epoch) + 1), alpha=index/num_epochs)
    # plt.legend()
    # plt.show()
    
    # for index, epoch in enumerate(test):
    plt.plot(x_test, y_test)
    plt.plot(x_test, y_preds)
    # plt.plot(x_test, epoch, label="Epoch: " + str(test.index(epoch) + 1), alpha=index/num_epochs)
    plt.legend()
    plt.show()


    plt.plot(epoch_loss)
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.title("Loss per epoch")
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    main()