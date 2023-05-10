import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.models import Sequential
import numpy as np
from matplotlib import pyplot as plt
import pickle
import sys


''' Create denoising autoencoder model that uses convolutions '''
def createDenoisingAutoencoder(input_shape, should_log = False):

    model = Sequential()

    model.add(layers.Conv2D(filters = 16, kernel_size = 3, strides = 1, padding='same', activation = 'relu', input_shape = input_shape))
    model.add(layers.BatchNormalization()) # H, W, 16
    model.add(layers.Conv2D(filters = 16, kernel_size = 3, strides = 2, padding='same', activation = 'relu'))
    model.add(layers.Conv2D(filters = 32, kernel_size = 3, strides = 1, padding='same', activation = 'relu'))
    model.add(layers.BatchNormalization()) # H/2, W/2, 32
    model.add(layers.Conv2DTranspose(32, kernel_size = 3, activation="relu", strides=2, padding="same"))
    model.add(layers.Conv2D(filters = 16, kernel_size = 3, strides = 1, padding='same', activation = 'relu'))
    model.add(layers.BatchNormalization()) # H, W, 16
    model.add(layers.Conv2D(filters = input_shape[2], kernel_size = 1, strides = 1, padding='same', activation = 'sigmoid'))

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate = 0.01, decay_steps = 10000, decay_rate = 0.9)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule), metrics=['accuracy'], loss='mean_squared_error')

    if should_log:
        model.summary()

    return model


''' Function to load CIFAR10 images that have been attacked, the same unattacked images, and the true class labels '''
def loadAttackData(filename):
    with open(filename, 'rb') as fin:
        X_attacked = pickle.load(fin)
        X_true = pickle.load(fin)
        y = pickle.load(fin)
    return X_attacked, X_true, y


'''
Function to plot a given image; credit https://github.com/Hyperparticle/one-pixel-attack-keras/blob/master/helper.py
'''
def plot_image(image, label_true=None, class_names=None, label_pred=None, save_img_filename=None):
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

    if(save_img_filename == None):
        plt.show()  # Show the plot
    else:
        plt.savefig(save_img_filename)  # Save the plot to the given file name


if __name__ == '__main__':

    use_existing_autoencoder = False

    if(len(sys.argv) > 1):
        cmd_arg = sys.argv[1]
        if(cmd_arg.lower() == '--use_existing_autoencoder'):
            use_existing_autoencoder = True
        else:
            print('Usage: python denoising_autoencoder.py --use_existing_autoencoder[optional]')
            exit()

    # Obtain attacked images, unattacked images, and the true class labels; split into train and test data
    X_attacked, X_unattacked, y = loadAttackData('attacked.imgs')
    X_attacked = np.asarray(X_attacked) / 255   # Scale RGB values between 0 and 1.0
    X_unattacked = np.asarray(X_unattacked) / 255   # Scale RGB values between 0 and 1.0
    X_attacked_train = X_attacked[0:int(len(X_attacked)/2)]
    X_attacked_test = X_attacked[int(len(X_attacked)/2):]
    X_unattacked_train = X_unattacked[0:int(len(X_unattacked)/2)]
    X_unattacked_test = X_unattacked[int(len(X_unattacked)/2):]
    y_train = np.asarray(y[0:int(len(y)/2)])
    y_test = np.asarray(y[int(len(y)/2):])

    # Load altered ResNet trained on CIFAR10 dataset using transfer learning
    resnet_model = models.load_model('./trainedModel', compile=False)
    resnet_model.compile()

    # Verify we got at least one attacked image
    if(len(X_attacked) < 1):
        print('Could not load attacked images')
        exit()

    # Create and train or load in existing denoising autoencoder model
    if not use_existing_autoencoder:
        den_autoencoder = createDenoisingAutoencoder(X_attacked[0].shape)
        den_autoencoder.fit(X_attacked_train, X_unattacked_train, epochs = 1000, batch_size = 256, shuffle = True, validation_data = (X_attacked_test, X_unattacked_test))
        den_autoencoder.save('./trainedDenAutoencoder')
    else:
        den_autoencoder = models.load_model('./trainedDenAutoencoder')

    # Show an image for testing
    plot_image(X_attacked_test[50] * 255, save_img_filename='attacked_img_test.png')
    plot_image(X_unattacked_test[50] * 255, save_img_filename='unattacked_img_test.png')
    plot_image(den_autoencoder.predict(X_attacked_test)[50] * 255, save_img_filename='denoised_img_test.png')