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
        print("delta = " + str(wtimesdelta) + " * " + str(self.activationderivative()) + " = " + str(self.delta))
        return self.weights * self.delta
    
    #Simply update the weights using the partial derivatives and the leranring weight
    def updateweight(self):
        print("first new weight -= " + str(self.lr) + " * " + str(self.delta) + " * " + str(self.input[0]) + " = " + str(self.lr * self.delta * self.input[0]))
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


#A convolutional layer 
class ConvolutionalLayer:
    #initialize with the number of kernels in the layer, the size of the kernel (assume it
    #is square), the activation function for all the neurons in the layer, the dimension of
    #the inputs, the learning rate, and a vector of weights (or else initilize randomly)
    def __init__(self, numOfKernels, kernelSize, activation, inputDim, lr, weights=None):
        self.numOfKernels = numOfKernels
        self.kernelSize = kernelSize
        self.activation = activation
        self.inputDim = inputDim
        self.lr = lr
        self.weights = weights

        # Find number of weights per kernel (not including bias)
        self.numWeightsPerKernel = (kernelSize ** 2) * inputDim[2]

        # Weights has 4 dimensions (w,h,input_channels,kernels)
        # Biases has 1 dimension (kernels)

        if weights is None:
            # Init all weights randomly if necessary
            self.weights = np.random.rand(kernelSize, kernelSize, inputDim[2], numOfKernels)
            self.biases = np.random.rand(numOfKernels)
        else:
            # Otherwise, interpret given vector of weights using numpy.reshape and last numOfKernels elements as biases
            self.weights = weights[:-numOfKernels].reshape(kernelSize, kernelSize, inputDim[2], numOfKernels)
            self.biases = weights[-numOfKernels:]

        # Create neuron objects in numpy array of all neurons in layer.
        # Arrange neurons in a 3D matrix  (w,h,channels).
        self.neurons = np.array([[[Neuron(activation, self.weights.size, lr, np.concatenate((self.weights[:,:,:,kernel_ind].ravel(), self.biases[kernel_ind:(kernel_ind+1)]))) for kernel_ind in range(numOfKernels)] for _ in range(inputDim[1] - kernelSize + 1)] for _ in range(inputDim[0] - kernelSize + 1)])

    #calculate the output of all the neurons in the layer and return them as a 3D matrix keyed on row, col, and channel
    def calculate(self, input):
        return np.array([[[self.neurons[width_ind, height_ind, channel_ind].calculate(input[width_ind : width_ind + self.kernelSize, height_ind : height_ind + self.kernelSize, : ].flatten()) for channel_ind in range(self.neurons.shape[2])] for height_ind in range(self.neurons.shape[1])] for width_ind in range(self.neurons.shape[0])])

    #given the next layer's w*delta, should run through the neurons calling 
    #calcpartialderivative() for each (with the correct value), sum up its own w*delta, and then 
    #update the weights (using the updateweight() method). I should return the sum of w*delta.          
    def calcwdeltas(self, wtimesdelta):

        # First calculate the weights times delta for every neuron in the layer
        wtimesdeltas = np.zeros((self.neurons.shape[0], self.neurons.shape[1], self.neurons.shape[2], self.numWeightsPerKernel))
        for neuron_o_ind, neuron_o in np.ndenumerate(self.neurons):
            wtimesdeltas[neuron_o_ind] = neuron_o.calcpartialderivative(wtimesdelta[neuron_o_ind])
            # neuron_o.updateweight()

        # Then update the weights for each kernel
        del_E_del_ws = np.zeros(self.weights.shape).tolist()
        for kernel_ind in range(self.numOfKernels):
            for row_o_ind in range(self.neurons.shape[0]):
                for col_o_ind in range(self.neurons.shape[1]):
                    for kernel_row_ind in range(self.kernelSize):
                        for kernel_col_ind in range(self.kernelSize):
                            for chan_x_ind in range(self.inputDim[2]):
                                del_E_del_ws[kernel_ind * self.numWeightsPerKernel + kernel_row_ind * self.kernelSize * self.inputDim[2] + kernel_col_ind * self.inputDim[2] + chan_x_ind] += self.neurons[row_o_ind, col_o_ind, kernel_ind].delta * self.neurons[row_o_ind, col_o_ind, kernel_ind].input[kernel_row_ind * self.kernelSize * self.inputDim[2] + kernel_col_ind * self.inputDim[2] + chan_x_ind]
                    
                    # Also have to do biases
                    del_E_del_ws[(kernel_ind+1) * self.numWeightsPerKernel - 1] += self.neurons[row_o_ind, col_o_ind, kernel_ind].delta

        # Do the shared weights update
        self.weights -= self.lr * np.asarray(del_E_del_ws)

        # Then sum the correct w times deltas together for backpropagation
        wtimesdeltaSum = np.zeros(self.inputDim).tolist()
        for row_x_ind in range(self.inputDim[0]):
            for col_x_ind in range(self.inputDim[1]):
                for chan_x_ind in range(self.inputDim[2]):
                    for kernel_row_ind in range(self.kernelSize):
                        for kernel_col_ind in range(self.kernelSize):
                            for kernel_ind in range(self.numOfKernels):
                                # print("\n\n")
                                # print(kernel_row_ind)
                                # print(self.inputDim[2])
                                # print(kernel_row_ind * self.kernelSize * self.inputDim[2] + kernel_col_ind * self.inputDim[2] + chan_x_ind)
                                row_o_ind = row_x_ind - kernel_row_ind
                                col_o_ind = col_x_ind - kernel_col_ind
                                if row_o_ind >= 0 and row_o_ind < wtimesdeltas.shape[0] and col_o_ind >= 0 and col_o_ind < wtimesdeltas.shape[1]:
                                    wtimesdeltaSum[row_x_ind][col_x_ind][chan_x_ind] += wtimesdeltas[row_o_ind, col_o_ind, kernel_ind, kernel_row_ind * self.kernelSize * self.inputDim[2] + kernel_col_ind * self.inputDim[2] + chan_x_ind]

        return np.asarray(wtimesdeltaSum)

        
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
            return -2 * (y-yp)
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

    # Add a layer to the neural network of a given type with given parameters and weights
    def addLayer(self, layer_type, layer_params=None, weights=None):
        
        valid_layer_types = ("FullyConnected", "Convolutional", "MaxPooling", "Flatten")

        if layer_type not in valid_layer_types:
            raise Exception("addLayer: layer_type of " + str(layer_type) + " is not valid")

        # Make FullyConnected layer
        if layer_type == valid_layer_types[0]:
            if layer_params == None or len(layer_params) != 2:
                raise Exception("addLayer: FullyConnected layer expects two arguments: number of neurons and activation")

            layer = FullyConnected(layer_params[0], layer_params[1], self.inputSize[0], self.lr, weights)
            self.inputSize = (layer.numOfNeurons,)
        
        # Make Convolutional layer
        elif layer_type == valid_layer_types[1]:
            if layer_params == None or len(layer_params) != 3:
                raise Exception("addLayer: Convolutional layer expects three arguments: number of kernels, kernel size, and activation")

            layer = ConvolutionalLayer(layer_params[0], layer_params[1], layer_params[2], self.inputSize, self.lr, weights)
            self.inputSize = layer.neurons.shape

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

        # Add layer to NeuralNetwork's list of layers
        self.layers.append(layer)



if __name__=="__main__":

    if (len(sys.argv)<2):
        print('a good place to test different parts of your code')

        img = np.reshape(np.linspace(0, 0.2, 1024), (16,16,4))
        output = np.reshape(np.linspace(0, 0.2, 200), (10,10,2))

        nn = NeuralNetwork((16, 16, 4), 0, 0.5)
        # nn.addLayer("Convolutional", (3, 5, 1), np.full(303, 0.2))
        # nn.addLayer("Convolutional", (2, 3, 1), np.full(56, 0.3))
        nn.addLayer("Convolutional", (3, 5, 1), np.linspace(-0.1, 0.202, 303))
        nn.addLayer("Convolutional", (2, 3, 1), np.linspace(-0.2, 0.35, 56))
        print(np.around(nn.calculate(img), 5))
        # nn.train(img, output)
        # print(nn.calculate(img))
        # print(nn.layers[0].weights)



    elif (sys.argv[1]=='example1'):
        print('run example1')
        

    elif(sys.argv[1]=='example2'):
        print('run example2')
        

    elif(sys.argv[1]=='example3'):
        print('run example3')