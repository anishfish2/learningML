import numpy as np

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

        This function calculates the Mean squared error of actual vs. predicted values. It includes a .5 factor to simplify the derivative
        
        :param predicted: Value(s) output by forward pass of model
        :param real: Actual value(s) of function
    """
    return .5 * np.mean((predicted - actual) ** 2)
