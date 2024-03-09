import numpy as np
import matplotlib.pyplot as plt

def he_initializer(lenInput):
    """
        He Initialization Function

        This function utilizes the He initialization method to instantiate node values

        :param lenInput: Number of Input Layer Neurons
        """
    
    stddev = np.sqrt(2 / lenInput) 
    return np.random.normal(0, stddev, size = lenInput)

def ReLU(x):
    """
        Standard implementation of ReLU

        This function returns the max of x and 0
        
        :param x: Value to apply ReLU
    """
    return np.maximum(0, x)


def ReLUD(x):
    """
        Derivative of ReLU

        This function returns the derivative of the ReLU function
        
        :param x: Value to apply ReLU Derivative
    """
    return np.where(x > 0, 1, 0)

def sigmoid(x):
    """
        Standard implementation of Sigmoid

        This function maps values between 0 and 1
        
        :param x: Value to apply Sigmoid
    """
    return 1 / (1 + np.exp(-x))

def sigmoidD(x):
    """
        Derivative of Sigmoid

        This function returns the derivative of the Sigmoid function
        
        :param x: Value to apply Sigmoid Derivative
    """
    sig = sigmoid(x)
    return sig * (1 - sig)

def tanh(x):
    """
    Hyperbolic tangent activation function
    
    This function returns tanh(x) 
    :param x: Input value(s)
    """
    return np.tanh(x)

def tanhD(x):
    """
    Derivative of the hyperbolic tangent activation function
    
    This function returns a derivative of tanh(x)

    :param x: Input value(s)
    
    """
    tanh_x = np.tanh(x)
    return 1 - tanh_x ** 2

def MSE(predicted, actual):
    """
        Standard implementation of Mean Squared Error

        This function calculates the Mean squared of actual vs. predicted values
        
        :param predicted: Value(s) output by forward pass of model
        :param real: Actual value(s) of function
    """
    return np.mean((predicted - actual) ** 2)


class Layer:
     def __init__(self, biases, weights, activation, activationD, numNodes, index):
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
        
        self.biases = biases
        self.weights = weights
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

        #Create the input layer with no biases, no weights, no activation function, and index 0
        inputLayer = Layer(np.zeros((lenInput)), np.identity((lenInput)), None, None, lenInput, 0)

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

        #Initialize random biases with near zero values
        biases = np.random.uniform(-0.01, 0.01, size=numNodes)

        #Initialize weights using He initialization method
        weights = np.zeros((self.Network[-1].numNodes, numNodes))
        for node in range(numNodes):
            weights[:, node] = he_initializer(self.Network[-1].numNodes)

        #Create hidden layer with random biases, random weights, given size and activation function, and assign index
        hiddenLayer = Layer(biases, weights, activation, activationD, numNodes, len(self.Network))
       
        #Add to network
        self.Network.append(hiddenLayer)

    def addOutput(self, lenOutput, activation, activationD):
        """
        Add Ouput Layer to Neural Network

        This function adds a hidden layer to the neural network.

        :param lenOutput: Number of Output Neurons
        :param activation: Function of Node Activation function
        """

        #Initialize random biases with near zero values
        biases = np.random.uniform(-0.01, 0.01, size=lenOutput)

        #Initialize weights using He initialization method
        weights = np.zeros((self.Network[-1].numNodes, lenOutput))
        for node in range(lenOutput):
            weights[:, node] = he_initializer(self.Network[-1].numNodes)
        #Create hidden layer with random biases, random weights, given size and activation function, and assign index
        outputLayer = Layer(biases, weights, activation, activationD, lenOutput, len(self.Network))
       
        #Add to network
        self.Network.append(outputLayer)
        
    def feedForward(self):
        """
        Run 1 forward pass through network

        This function takes values, runs them through current state of network and returns output values

        :param inputs: Input values to Input Layer
        """
        self.Network[0].values = self.input_values
        for i in range(1, len(self.Network)):
            layer = self.Network[i]
            prev_layer = self.Network[i-1]
            layer.values = np.dot(prev_layer.values, layer.weights) + layer.biases
            if layer.activation:
                layer.values = layer.activation(layer.values)
            
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
        gradient = 2 * error

        # Backpropagate the error through the layers
        for i in range(len(self.Network) - 1, 0, -1):
            current_layer = self.Network[i]
            prev_layer = self.Network[i - 1]
            # Compute the gradients for weights and biases
            if current_layer.activationD is not None:
                activation_derivative = current_layer.activationD(current_layer.values)
                gradient = gradient * activation_derivative

            else:
                gradient = gradient

            # Compute weight and bias gradients
            weight_gradient = np.outer(gradient, prev_layer.values).T
            bias_gradient = gradient

            # Update weights and biases using gradients and learning rate

            current_layer.weights -= self.learningRate * weight_gradient
            current_layer.biases -= self.learningRate * bias_gradient
            if i > 1:
                error = np.dot(current_layer.weights, gradient)[0]
            
def testFunction(x):
    return np.sin(x)

def main():

    num_vals = 10000
    x = [[i] for i in np.linspace(-5, 5, num=num_vals)]
    print(x)
    y = testFunction(x)
    NN = NeuralNetwork(1, 0.00000001)
    NN.addHidden(20, ReLU, ReLUD)
    NN.addHidden(10, ReLU, ReLUD)
    NN.addOutput(1, tanh, tanhD)

    for i in range(num_vals):
        NN.setInput(x[i])
        NN.setInputTest(y[i])

        prediction = NN.feedForward()[0]
        true_val = y[i]
        NN.backProp(prediction, true_val)

    x_test = [[i] for i in np.linspace(-5, 5, num=num_vals)]
    y_test = testFunction(x_test)
    
    test = []
    for i in range(num_vals):
        NN.setInput(x_test[i])
        prediction = NN.feedForward()[0]
        test.append(prediction)

    plt.plot(x_test, y_test)
    plt.plot(x_test, test)
    plt.show()
if __name__ == "__main__":
    main()