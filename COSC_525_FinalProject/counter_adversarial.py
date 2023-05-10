import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.models import Sequential
import numpy as np
from matplotlib import pyplot as plt
import pickle
import sys
import os
import math


# Define possible command line arguments as global tuples
REQ_OPTIONS = ('--architecture',)
OPT_OPTIONS = ('--train',)
REQUIRES_ARG = ('--architecture',)
ARCHITECTURES = ('fully_conv', 'fully_conn')


''' Create denoising autoencoder model that uses convolutions '''
def createFullyConvDenoisingAutoencoder(input_shape, should_log = False):

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

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate = 0.01, decay_steps = 500, decay_rate = 0.5)
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
Function to plot a grid of images given in a numpy array; credit https://stackoverflow.com/questions/20038648/writting-a-file-with-multiple-images-in-a-grid (user James Bond)
'''
def plot_image_grid(image_arr, save_grid_filename=None):
    plt.clf()

    result_figsize_resolution = 40 # 1 = 100px

    images_count = image_arr.shape[0]
    print('Images count: ', images_count)

    # Calculate the grid size:
    grid_size = math.ceil(math.sqrt(images_count))

    # Create plt plot:
    _, axes = plt.subplots(grid_size, grid_size, figsize=(result_figsize_resolution, result_figsize_resolution))

    # Plot every image in the grid
    current_image_number = 0
    for image in image_arr:
        x_position = current_image_number % grid_size
        y_position = current_image_number // grid_size
        axes[x_position, y_position].imshow(image.astype(np.uint8))
        current_image_number += 1

    if(save_grid_filename == None):
        plt.show()  # Show the plot
    else:
        plt.savefig(save_grid_filename)  # Save the plot to the given file name

'''
Function to plot a given image; credit https://github.com/Hyperparticle/one-pixel-attack-keras/blob/master/helper.py
'''
def plot_image(image, label_true=None, class_names=None, label_pred=None, save_img_filename=None):
    plt.clf()

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


''' Print a single option to the user inform them how to use this option '''
def print_option(option):
    if option in REQUIRES_ARG:
        print('    ' + option + ' [required argument]')
    else:
        print('    ' + option)


''' Print out how to use this script to the user '''
def print_usage():
    print('USAGE: python counter_adversarial.py [options...]')
    print('REQUIRED OPTIONS:')
    for option in REQ_OPTIONS:
        print_option(option)
    print('\nOPTIONAL OPTIONS:')
    for option in OPT_OPTIONS:
        print_option(option)
    exit()


if __name__ == '__main__':

    architecture = None
    should_train = False

    # Process command line arguments
    req_options_met = [False] * len(REQ_OPTIONS)
    cmd_arg_it = 1
    while(cmd_arg_it < len(sys.argv)):
        cmd_arg = sys.argv[cmd_arg_it].lower()

        # Process required options to make sure we have all required options
        if cmd_arg in REQ_OPTIONS:
            req_option_ind = REQ_OPTIONS.index(cmd_arg)
            req_options_met[req_option_ind] = True
        
        # Ensure command line argument is valid at all
        elif cmd_arg not in OPT_OPTIONS:
            print_usage()

        # Use options by storing appropriate values in corresponding variables
        
        # Obtain and validate desired model architecture
        if cmd_arg == '--architecture':
            if(cmd_arg_it + 1 == len(sys.argv)):
                print_usage()
            cmd_arg_it += 1
            architecture = sys.argv[cmd_arg_it]
            if architecture not in ARCHITECTURES:
                print('AVAILABLE ARCHITECTURE ARGUMENTS:')
                for a in ARCHITECTURES:
                    print('    ' + a)
                exit()

        # Obtain if we should train the target model
        elif cmd_arg == '--train':
            should_train = True

        cmd_arg_it += 1

    # Ensure all required command line arguments are met
    for b in req_options_met:
        if not b:
            print_usage()

    # Obtain attacked images, unattacked images, and the true class labels; split into train and test data
    X_attacked, X_unattacked, y = loadAttackData('attacked.imgs')
    X_attacked = np.asarray(X_attacked) / 255   # Scale RGB values between 0 and 1.0
    X_unattacked = np.asarray(X_unattacked) / 255   # Scale RGB values between 0 and 1.0
    X_attacked_test = X_attacked[0:int(len(X_attacked)/3)]
    X_attacked_train = X_attacked[int(len(X_attacked)/3):]
    X_unattacked_test = X_unattacked[0:int(len(X_unattacked)/3)]
    X_unattacked_train = X_unattacked[int(len(X_unattacked)/3):]
    y_test = np.asarray(y[0:int(len(y)/3)])
    y_train = np.asarray(y[int(len(y)/3):])

    # Load altered ResNet trained on CIFAR10 dataset using transfer learning
    resnet_model = models.load_model('./trainedModel', compile=False)
    resnet_model.compile()

    # Verify we got at least one attacked image
    if(len(X_attacked) < 1):
        print('Could not load attacked images')
        exit()

    # Create and train or load in existing model
    if should_train:
        if(architecture == 'fully_conv'):
            model = createFullyConvDenoisingAutoencoder(X_attacked[0].shape)
        
        model.fit(X_attacked_train, X_unattacked_train, epochs = 1000, batch_size = 256, shuffle = True, validation_data = (X_attacked_test, X_unattacked_test))
        model.save('./trained_' + architecture)
    else:
        if not os.path.exists('./trained_' + architecture):
            print('Existing model for architecture ' + architecture + ' not found; must be trained using --train')
            exit()
        model = models.load_model('./trained_' + architecture)

    # Save a grid of test images for each of the following image types: attacked, unattacked, and denoised
    plot_image_grid(X_attacked_test[:100] * 255, save_grid_filename='attacked_imgs_test_' + architecture + '.png')
    plot_image_grid(X_unattacked_test[:100] * 255, save_grid_filename='unattacked_imgs_test_' + architecture + '.png')
    plot_image_grid(model.predict(X_attacked_test)[:100] * 255, save_grid_filename='denoised_attacked_imgs_test_' + architecture + '.png')
    plot_image_grid(model.predict(X_unattacked_test)[:100] * 255, save_grid_filename='denoised_unattacked_imgs_test_' + architecture + '.png')