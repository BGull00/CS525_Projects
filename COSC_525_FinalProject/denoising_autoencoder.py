import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import numpy as np
from matplotlib import pyplot as plt
import pickle

''' Create denoising autoencoder model that uses convolutions '''
def createDenoisingAutoencoder(input_shape):

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

    model.compile(optimizer='adam', metrics=['accuracy'], loss='mean_squared_error')
    model.summary()

    return model

''' Function to load CIFAR10 images that have been attacked, the same unattacked images, and the true class labels '''
def loadAttackData(filename):
    with open(filename, 'rb') as fin:
        X_attacked = pickle.load(fin)
        X_true = pickle.load(fin)
        y = pickle.load(fin)
    return X_attacked, X_true, y

if __name__ == '__main__':

    # Obtain attacked images, unattacked images, and the true class labels
    X_attacked, X_unattacked, y = loadAttackData('attacked.imgs')

    # Verify we got at least one attacked image
    if(len(X_attacked) < 1):
        print('Could not load attacked images')
        exit()

    # Create denoising autoencoder model
    den_autoencoder = createDenoisingAutoencoder(X_attacked[0].shape)

    # plt.imshow(X_attacked[50].astype(np.uint8))
    # plt.show()