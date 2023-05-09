from tensorflow.keras.datasets import cifar10
from tensorflow.keras import models
import tensorflow as tf
from art.attacks.evasion import PixelAttack
from art.estimators.classification import TensorFlowV2Classifier
import matplotlib.pyplot as plt
import numpy as np
import pickle

'''
Function gotten from https://github.com/Hyperparticle/one-pixel-attack-keras/blob/master/helper.py
'''
def plot_image(image, label_true=None, class_names=None, label_pred=None):
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]

    plt.grid()
    plt.imshow(image.astype(np.uint8))

    # Show true and predicted classes
    if label_true is not None and class_names is not None:
        labels_true_name = class_names[label_true]
        if label_pred is None:
            xlabel = "True: " + labels_true_name
        else:
            # Name of the predicted class
            labels_pred_name = class_names[label_pred]
            xlabel = "True: " + labels_true_name + "\nPredicted: " + labels_pred_name
        # Show the class on the x-axis
        plt.xlabel(xlabel)
    plt.xticks([])  # Remove ticks from the plot
    plt.yticks([])
    plt.show()  # Show the plot

def loadAttackData(filename):
    with open(filename, 'rb') as fin:
        X_attacked = pickle.load(fin)
        X_true = pickle.load(fin)
        y = pickle.load(fin)
    return X_attacked, X_true, y
    #return np.array(X_attacked), np.array(X_true), y
    

if __name__ == '__main__':
    #Load data and Model
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    model = models.load_model('./trainedModel')

    #Create A.R.T. Classifier and Pixel Attack
    classifier = TensorFlowV2Classifier(model=model, nb_classes=10, input_shape=(32,32,3), clip_values=(0,255))
    attack = PixelAttack(classifier, verbose=True, th=1, max_iter=5)

    attack_data, X_data, y = loadAttackData('attacked.imgs')
    
    for i in range(len(attack_data), len(X_train)):
        X_attacked = attack.generate(x=np.array([X_train[i]]))
        attack_data.append(X_attacked[0])
        X_data.append(X_train[i])
        with open('attacked.imgs', 'wb') as fout:
            pickle.dump(attack_data, fout)
            pickle.dump(X_data, fout)
            pickle.dump(y_train[:i+1], fout)
        if i % 99 == 0 and i > 0:    
            model.evaluate(np.array(attack_data), y_train[:i+1])
            model.evaluate(np.array(X_data), y_train[:i+1])