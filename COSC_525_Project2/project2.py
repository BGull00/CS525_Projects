import numpy as np
import sys
"""
For this entire file there are a few constants:
activation:
0 - linear
1 - logistic (only one supported)
loss:
0 - sum of square errors
1 - binary cross entropy
"""


# A class which represents a single neuron
class Neuron:
    #initilize neuron with activation type, number of inputs, learning rate, and possibly with 
    #set weights
    def __init__(self,activation, input_num, lr, weights=None):
        self.activation = activation
        self.input_num = input_num
        self.lr = lr
        self.weights = weights
        if weights is None:
            self.weights = np.random.rand(input_num+1)
        
    #This method returns the activation of the net
    def activate(self,net):
        if self.activation == 0:
            # Linear activation function
            return net
        else:
            # Logistic activation function
            return 1/(1+np.exp(-net))
        
    #Calculate the output of the neuron should save the input and output for back-propagation.   
    def calculate(self,input):
        self.input = np.append(input, 1)
        self.net = np.sum(self.input*self.weights)
        self.output = self.activate(self.net)
        return self.output

    #This method returns the derivative of the activation function with respect to the net   
    def activationderivative(self):
        if self.activation == 0:
            # Derivative of linear activation function
            return 1
        else:
            # Derivative of logistic activation function
            return self.output * (1 - self.output)

    #This method calculates the partial derivative for each weight and returns the delta*w to 
    #be used in the previous layer
    def calcpartialderivative(self, wtimesdelta):
        self.delta = wtimesdelta * self.activationderivative()
        return self.weights * self.delta
    
    #Simply update the weights using the partial derivatives and the leranring weight
    def updateweight(self):
        self.weights -= self.lr * self.delta * self.input

        
#A fully connected layer        
class FullyConnected:
    #initialize with the number of neurons in the layer, their activation,the input size, the 
    #leraning rate and a 2d matrix of weights (or else initilize randomly)
    def __init__(self,numOfNeurons, activation, input_num, lr, weights=None):
        self.numOfNeurons = numOfNeurons
        self.activation = activation
        self.input_num = input_num
        self.lr = lr
        self.weights = weights

        # weights is a 2D matrix; first index is neuron, second index is weight

        # Init weights randomly if necessary
        if weights is None:
            self.weights = np.random.rand(numOfNeurons, input_num + 1)
            bias = np.random.rand(1)
            self.weights[:,-1] = bias[0]

        # Create neuron objects in numpy array of all neurons in layer
        self.neurons = np.array([Neuron(activation, input_num, lr, self.weights[neuronInd]) for neuronInd in range(numOfNeurons)])
        
    #calcualte the output of all the neurons in the layer and return a vector with those values 
    #(go through the neurons and call the calcualte() method)      
    def calculate(self, input):
        return np.array([neuron.calculate(input) for neuron in self.neurons])
        
            
    #given the next layer's w*delta, should run through the neurons calling 
    #calcpartialderivative() for each (with the correct value), sum up its ownw*delta, and then 
    #update the wieghts (using the updateweight() method). I should return the sum of w*delta.          
    def calcwdeltas(self, wtimesdelta):
        wtimesdeltaSum = 0
        for neuronInd, neuron in enumerate(self.neurons):
            wtimesdeltaSum += neuron.calcpartialderivative(wtimesdelta[neuronInd])
            neuron.updateweight()

        return wtimesdeltaSum
           
        
#An entire neural network        
class NeuralNetwork:
    #initialize with the input size, loss function, and learning rate
    def __init__(self, inputSize, loss, lr):
        self.inputSize = inputSize
        self.loss = loss
        self.lr = lr

        # Start with a list of no layers
        self.layers = []
    
    #Given an input, calculate the output (using the layers calculate() method)
    def calculate(self,input):
        prevLayerOutput = input

        # Loop through every layer in NN and use output of previous layer as input to next layer
        for layer in self.layers:
            prevLayerOutput = layer.calculate(prevLayerOutput)

        # Return output of NN
        return prevLayerOutput

    #Given a predicted output and ground truth output simply return the loss (depending on the loss function)
    def calculateloss(self,yp,y):
        if self.loss == 0:
            # Sum of square errors loss function
            return np.sum(np.square(yp-y)) / len(yp)
        else:
            # Binary cross entropy loss function
            return np.sum((y*np.log(yp) + (1 - y)*np.log(1 - yp))) / -len(yp)
    
    #Given a predicted output and ground truth output simply return the derivative of the loss 
    #(depending on the loss function)        
    def lossderiv(self,yp,y):
        if self.loss == 0:
            # Derivative of sum of square errors loss function
            return -(y-yp)
        else:
            # Derivative of binary cross entropy loss function
            return -((y/yp) - ((1-y)/(1-yp)))
    
    #Given a single input and desired output preform one step of backpropagation (including a 
    #forward pass, getting the derivative of the loss, and then calling calcwdeltas for layers 
    # with the right values         
    def train(self,x,y):

        # Feed forward calculation
        output = self.calculate(x)
        
        # Backpropagation
        prevWtimesdelta = self.lossderiv(output, y)
        for layer in reversed(self.layers):
            prevWtimesdelta = layer.calcwdeltas(prevWtimesdelta)

    # Add a layer to the neural network
    def addLayer(self, layer_type, layer_params=None, weights=None):
        
        valid_layer_types = ("FullyConnected", "Convolutional", "MaxPooling", "Flatten")

        if layer_type not in valid_layer_types:
            raise Exception("addLayer: layer_type of " + str(layer_type) + " is not valid")

        # Make FullyConnected layer
        if layer_type == valid_layer_types[0]:
            if layer_params == None or len(layer_params) != 2:
                raise Exception("addLayer: FullyConnected layer expects two arguments: number of neurons and activation")

            layer = FullyConnected(layer_params[0], layer_params[1], self.inputSize, self.lr, weights)
            self.inputSize = (layer.numOfNeurons)
        
        # Make Convolutional layer
        elif layer_type == valid_layer_types[1]:
            if layer_params == None or len(layer_params) != 3:
                raise Exception("addLayer: Convolutional layer expects three arguments: number of kernels, kernel size, and activation")

            #layer = ConvolutionalLayer(layer_params[0], layer_params[1], layer_params[2], self.inputSize, self.lr, weights)
            #self.inputSize = 

        # Make MaxPooling layer
        elif layer_type == valid_layer_types[2]:
            if layer_params == None or len(layer_params) != 1:
                raise Exception("addLayer: MaxPooling layer expects one argument: kernel size")

            #layer = MaxPoolingLayer(layer_params[0], self.inputSize)
            #self.inputSize = 

        # Make Flatten layer
        elif layer_type == valid_layer_types[3]:
            if layer_params != None:
                raise Exception("addLayer: Flatten layer expects no arguments")

            #layer = FlattenLayer(self.inputSize)
            #self.inputSize = 

        self.layers.append(layer)



if __name__=="__main__":

    if (len(sys.argv)<2):
        print('a good place to test different parts of your code')
        

    elif (sys.argv[1]=='example1'):
        print('run example1')
        

    elif(sys.argv[1]=='example2'):
        print('run example2')
        

    elif(sys.argv[1]=='example3'):
        print('run example3')