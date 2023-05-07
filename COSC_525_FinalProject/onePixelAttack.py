from tensorflow.keras.datasets import cifar10
from tensorflow.keras import models
import tensorflow as tf
from art.estimators.classification import TensorFlowV2Classifier
import matplotlib.pyplot as plt
import numpy as np
from art.attacks.evasion import PixelAttack

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

if __name__ == '__main__':

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    model = models.load_model('./trainedModel')
    model.evaluate(X_test, y_test)
    classifier = TensorFlowV2Classifier(model=model, nb_classes=10, input_shape=(32,32,3), clip_values=(0,255))
    attack = PixelAttack(classifier, verbose=True)
    X_test_adv = attack.generate(X_test)
    model.evaluate(X_test_adv, y_test)