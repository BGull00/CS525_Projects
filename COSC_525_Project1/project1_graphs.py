from project1 import NeuralNetwork
from matplotlib import pyplot as plt
import numpy as np

if __name__ == '__main__':
    
    '''
    losses = []
    lrs = [1, 0.8, 0.5, 0.2, 0.1, 0.05, 0.01]
    labels = []
    for lr in lrs:
        labels.append(str(lr))
        xData = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        yData = np.array([[0], [1], [1], [0]])

        nn = NeuralNetwork(2, np.array([2, 1]), 2, 1, 1, lr)
        loss_data = []
        for j in range(10000):
            dataInd = np.random.randint(4, size=1)[0]
            nn.train(xData[dataInd], yData[dataInd])
            loss = 0
            if j % 100 == 0:
                for k in range(len(xData)):
                    out = nn.calculate(xData[k])
                    loss += nn.calculateloss(out, yData[k])
                loss_data.append(loss/len(xData))
        losses.append(loss_data)
    for loss in losses:
        plt.plot(loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss of XOR Problem With Various Learning Rates')
    plt.legend(lrs, title="Learning Rates")
    plt.show()
    '''
    
    
    lrs = [1, 0.5, 0.1, 0.05, 0.01, 0.005]
    labels = []
    losses = []
    for lr in lrs:
        labels.append(str(lr))

        w = np.array([[[.15,.2,.35],[.25,.3,.35]],[[.4,.45,.6],[.5,.55,.6]]])
        x = np.array([0.05,0.1])
        y = np.array([0.01,0.99])

        # Build and train example neural net from class
        nn = NeuralNetwork(2, np.array([2, 2]), 2, 1, 0, lr, w)

        loss_data = []
        for i in range(1000):
            nn.train(x, y)
            loss_data.append(nn.calculateloss(nn.calculate(x), y))
        losses.append(loss_data)
    for loss in losses:
        plt.plot(loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss of Example From Class With Various Learning Rates')
    plt.legend(lrs, title="Learning Rates")
    plt.savefig("loss_graph_example.png")
    