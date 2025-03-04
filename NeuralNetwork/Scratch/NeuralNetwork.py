import numpy as np
import matplotlib.pyplot as plt
from loss_functions import *
from tqdm import tqdm

class LinearLayer:
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

        inputLayer = LinearLayer(None, None, None, lenInput, 0)

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

        matrix = np.zeros((numNodes, self.Network[-1].numNodes + 1))
        for node in range(numNodes):
            matrix[node, :-1] = he_initializer(self.Network[-1].numNodes)
        layer = LinearLayer(matrix, activation, activationD, numNodes, len(self.Network))

        #Add to network
        self.Network.append(layer)
        
    def feedForward(self, input):
        """
        Run 1 forward pass through network

        This function takes values, runs them through current state of network and returns output values

        :param inputs: Input values to Input Layer
        """
        self.Network[0].values = np.ravel(input)
        for i in range(1, len(self.Network)):
            layer = self.Network[i]
            prev_layer = self.Network[i-1]
            layer.values = np.append(prev_layer.values, 1) @ layer.weight_matrix.T  # Add bias
            if layer.activation:
                layer.values = layer.activation(layer.values)

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
                    layer.value_gradient_matrix = lossFunctionD(actual, predicted) * layer.activationD(layer.values)
                else:
                    layer.value_gradient_matrix = lossFunctionD(actual, predicted)
            else:
                if layer.activation:
                    layer.value_gradient_matrix = layer.activationD(layer.values) * (self.Network[index+1].weight_matrix[:, :-1].T @ self.Network[index + 1].value_gradient_matrix)
                else:
                    layer.value_gradient_matrix = self.Network[index+1].weight_matrix @ self.Network[index + 1].value_gradient_matrix
            layer.weight_gradient_matrix = np.reshape(layer.value_gradient_matrix, (-1, 1)) @ np.reshape(np.append(next_layer.values, 1), (1, -1))

            # Gradient Clipping to avoid exploding gradients
            max_grad = 5.0  
            layer.weight_gradient_matrix = np.clip(layer.weight_gradient_matrix, -max_grad, max_grad)
            layer.weight_matrix -= lr * layer.weight_gradient_matrix  # Update weights


        return loss

def testFunction(x):
    """
        Define a function to model in main function

        This function is used to model the function that the neural network will try to learn

        :param value: input value to function
    """
    return x ** 2

def main():
    num_vals = 1000
    num_epochs = 2000

    x = np.array([[i] for i in np.linspace(-1, 1, num_vals)])
    y = testFunction(x)

    NN = NeuralNetwork(1)
    NN.addLayer(16, leakyReLU, leakyReLUD)
    NN.addLayer(1, identity, identityD, True)

    epoch_loss = []

    # Create a single figure with 2 subplots:
    plt.ion()  # Turn on interactive mode so plt.pause() works
    fig, (ax_pred, ax_loss) = plt.subplots(2, 1, figsize=(8, 10))

    for epoch in tqdm(range(num_epochs)):
        total_loss = 0
        for i in range(num_vals):
            prediction = NN.feedForward(x[i])
            true_val = y[i]
            loss = NN.backProp(true_val, prediction, MSE, MSED, lr=0.001)
            total_loss += loss

        epoch_loss.append(total_loss)

        x_test = np.array([[i] for i in np.linspace(-5, 5, num_vals)])
        y_test = testFunction(x_test)
        y_preds = []
        for i in range(num_vals):
            y_preds.append(NN.feedForward(x_test[i])[0])

        ax_pred.cla()  # Clear old lines
        ax_pred.plot(x_test, y_test, label="True sin(x)")
        ax_pred.plot(x_test, y_preds, label=f"Epoch {epoch+1} Prediction")
        ax_pred.set_xlabel('x')
        ax_pred.set_ylabel('y')
        ax_pred.set_title('Network Approximation Over Epochs')
        ax_pred.legend()

        ax_loss.cla()
        ax_loss.plot(epoch_loss, label="Training Loss")
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss')
        ax_loss.set_title('Loss Over Time')
        ax_loss.legend()

        plt.pause(0.01) 

    # Leave the final figures open
    plt.ioff()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(epoch_loss)
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.title("Loss per Epoch")
    plt.show()

if __name__ == "__main__":
    main()
