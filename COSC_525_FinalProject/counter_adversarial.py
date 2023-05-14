import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.models import Sequential
from scipy.ndimage import gaussian_filter
import numpy as np
import visualkeras
from matplotlib import pyplot as plt
import pickle
import sys
import os
import math


# Define possible command line arguments as global tuples
REQ_OPTIONS = ('--architecture',)
OPT_OPTIONS = ('--train',)
REQUIRES_ARG = ('--architecture',)
ARCHITECTURES = ('fully_conv', 'fully_conn', 'gan', 'blur')


''' Create denoising autoencoder model that uses convolutions '''
def create_fully_conv_denoising_autoencoder(input_shape, should_log = False):

    model = Sequential()

    model.add(layers.Conv2D(filters = 16, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu', input_shape = input_shape))
    model.add(layers.BatchNormalization()) # H, W, 16
    model.add(layers.Conv2D(filters = 16, kernel_size = 3, strides = 2, padding = 'same', activation = 'relu'))
    model.add(layers.Conv2D(filters = 32, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu'))
    model.add(layers.BatchNormalization()) # H/2, W/2, 32
    model.add(layers.Conv2DTranspose(filters = 32, kernel_size = 3, strides = 2, padding = 'same', activation = "relu"))
    model.add(layers.Conv2D(filters = 16, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu'))
    model.add(layers.BatchNormalization()) # H, W, 16
    model.add(layers.Conv2D(filters = input_shape[2], kernel_size = 1, strides = 1, padding = 'same', activation = 'sigmoid'))

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate = 0.01, decay_steps = 500, decay_rate = 0.5)
    model.compile(optimizer = keras.optimizers.Adam(learning_rate = lr_schedule), loss = 'mean_squared_error')

    if should_log:
        model.summary()

    return model


''' Create denoising autoencoder model that uses fully connected layers '''
def create_fully_conn_denoising_autoencoder(input_shape, should_log = False):

    model = Sequential()

    input_size = 1
    for val in input_shape:
        input_size *= val

    model.add(layers.Input(shape = input_shape))
    model.add(layers.Flatten())
    model.add(layers.Dense(units = 128, activation = 'relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(units = input_size, activation = 'sigmoid'))
    model.add(layers.Reshape(input_shape))

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate = 0.001, decay_steps = 1000, decay_rate = 0.5)
    model.compile(optimizer = keras.optimizers.Adam(learning_rate = lr_schedule), loss = 'mean_squared_error')

    if should_log:
        model.summary()

    return model


''' Create generative adversarial denoising autoencoder model that uses fully convolutional layers in the
autoencoder followed by a discriminator that also uses convolutional layers '''
def create_GAN_denoising_autoencoder(input_shape, should_log = False):

    autoencoder = Sequential()
    autoencoder.add(layers.Conv2D(filters = 16, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu', input_shape = input_shape))
    autoencoder.add(layers.BatchNormalization()) # H, W, 16
    autoencoder.add(layers.Conv2D(filters = 16, kernel_size = 3, strides = 2, padding = 'same', activation = 'relu'))
    autoencoder.add(layers.Conv2D(filters = 32, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu'))
    autoencoder.add(layers.BatchNormalization()) # H/2, W/2, 32
    autoencoder.add(layers.Conv2DTranspose(filters = 32, kernel_size = 3, strides = 2, padding = 'same', activation = "relu"))
    autoencoder.add(layers.Conv2D(filters = 16, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu'))
    autoencoder.add(layers.BatchNormalization()) # H, W, 16
    autoencoder.add(layers.Conv2D(filters = input_shape[2], kernel_size = 1, strides = 1, padding = 'same', activation = 'sigmoid'))

    if should_log:
        autoencoder.summary()

    discriminator = Sequential()
    discriminator.add(layers.Conv2D(filters = 16, kernel_size = 3, strides = 1, padding = 'same', activation = layers.LeakyReLU(), input_shape = input_shape))
    discriminator.add(layers.BatchNormalization()) # H, W, 16
    discriminator.add(layers.Dropout(0.5))
    discriminator.add(layers.Conv2D(filters = 16, kernel_size = 3, strides = 2, padding = 'same', activation = layers.LeakyReLU()))
    discriminator.add(layers.Conv2D(filters = 32, kernel_size = 3, strides = 1, padding = 'same', activation = layers.LeakyReLU()))
    discriminator.add(layers.BatchNormalization()) # H/2, W/2, 32
    discriminator.add(layers.Dropout(0.5))
    discriminator.add(layers.Flatten())
    discriminator.add(layers.Dense(units = 1, activation = 'sigmoid'))

    if should_log:
        discriminator.summary()

    return autoencoder, discriminator


''' Function used to calculate loss values for generative adversarial denoising autoencoder model '''
def get_GAN_denoising_autoencoder_losses(unattacked_images, denoised_images, denoised_disc_output, unattacked_disc_output, mse, binary_cross_entropy):

    # Calculate autoencoder loss (sum of MSE and binary cross entropy from discriminator being fooled
    # into thinking denoised images are unattacked)
    mse_autoencoder_loss = mse(unattacked_images, denoised_images)
    discriminator_output_loss = binary_cross_entropy(tf.ones_like(denoised_disc_output), denoised_disc_output)
    autoencoder_loss = 75 * mse_autoencoder_loss + discriminator_output_loss

    # Calculate discriminator loss (sum of binary cross entropy from correctly classifying unattacked
    # images as unattacked and from correctly classifying denoised images as denoised)
    unattacked_discriminator_loss = binary_cross_entropy(tf.ones_like(unattacked_disc_output), unattacked_disc_output)
    denoised_discriminator_loss = binary_cross_entropy(tf.zeros_like(denoised_disc_output), denoised_disc_output)
    discriminator_loss = unattacked_discriminator_loss + denoised_discriminator_loss

    return autoencoder_loss, discriminator_loss


''' Function used by TensorFlow to train the generative adversarial denoising autoencoder for a single step;
credit: https://www.tensorflow.org/tutorials/generative/dcgan '''
@tf.function
def train_GAN_denoising_autoencoder_step(attacked_images, unattacked_images, autoencoder, discriminator, mse, binary_cross_entropy, autoencoder_optimizer, discriminator_optimizer):

    # Perform training step
    with tf.GradientTape() as autoencoder_tape, tf.GradientTape() as discriminator_tape:
        denoised_images = autoencoder(attacked_images, training = True)

        denoised_disc_output = discriminator(denoised_images, training = True)
        unattacked_disc_output = discriminator(unattacked_images, training = True)

        autoencoder_loss, discriminator_loss = get_GAN_denoising_autoencoder_losses(unattacked_images, denoised_images, denoised_disc_output, unattacked_disc_output, mse, binary_cross_entropy)

    gradients_of_discriminator = discriminator_tape.gradient(discriminator_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    gradients_of_autoencoder = autoencoder_tape.gradient(autoencoder_loss, autoencoder.trainable_variables)
    autoencoder_optimizer.apply_gradients(zip(gradients_of_autoencoder, autoencoder.trainable_variables))


''' Function used to train generative adversarial denoising autoencoder model;
credit: https://www.tensorflow.org/tutorials/generative/dcgan '''
def train_GAN_denoising_autoencoder(attacked_images_train, unattacked_images_train, attacked_images_val, unattacked_images_val, num_epochs, batch_size, autoencoder, discriminator):

    # Init loss functions from Keras
    mse = keras.losses.MeanSquaredError()
    binary_cross_entropy = keras.losses.BinaryCrossentropy()

    # Init Adam optimizers for both autoencoder and generator
    autoencoder_optimizer = keras.optimizers.Adam(learning_rate = 0.0001)
    discriminator_optimizer = keras.optimizers.Adam(learning_rate = 0.0001)

    # Calculate number of batches per epoch
    num_batches_per_epoch = int(attacked_images_train.shape[0] / batch_size)

    # Train for given number of epochs
    autoencoder_losses_train = []
    discriminator_losses_train = []
    autoencoder_losses_val = []
    discriminator_losses_val = []
    for epoch in range(num_epochs):

        print('Starting epoch ' + str(epoch+1) + '/' + str(num_epochs))
        
        # Train one step for each batch in a single epoch
        for _ in range(num_batches_per_epoch):

            # Randomly select batch of batch_size samples from training data
            batch_rand_img_inds = np.random.randint(0, attacked_images_train.shape[0], batch_size)
            attacked_images_batch = attacked_images_train[batch_rand_img_inds]
            unattacked_images_batch = unattacked_images_train[batch_rand_img_inds]

            # Perform training step on current batch of attacked and unattacked images
            train_GAN_denoising_autoencoder_step(attacked_images_batch, unattacked_images_batch, autoencoder, discriminator, mse, binary_cross_entropy, autoencoder_optimizer, discriminator_optimizer)
            
        # Calculate and print losses for the epoch for all training data
        denoised_images_train = autoencoder(attacked_images_train, training = False)
        denoised_disc_output_train = discriminator(denoised_images_train, training = False)
        unattacked_disc_output_train = discriminator(unattacked_images_train, training = False)
        autoencoder_loss_train, discriminator_loss_train = get_GAN_denoising_autoencoder_losses(unattacked_images_train, denoised_images_train, denoised_disc_output_train, unattacked_disc_output_train, mse, binary_cross_entropy)

        print('    autoencoder_loss_train: ' + str(float(autoencoder_loss_train)) + ' - discriminator_loss_train: ' + str(float(discriminator_loss_train)))
        autoencoder_losses_train.append(autoencoder_loss_train)
        discriminator_losses_train.append(discriminator_loss_train)

        # Calculate and print losses for the epoch for all validation data
        denoised_images_val = autoencoder(attacked_images_val, training = False)
        denoised_disc_output_val = discriminator(denoised_images_val, training = False)
        unattacked_disc_output_val = discriminator(unattacked_images_val, training = False)
        autoencoder_loss_val, discriminator_loss_val = get_GAN_denoising_autoencoder_losses(unattacked_images_val, denoised_images_val, denoised_disc_output_val, unattacked_disc_output_val, mse, binary_cross_entropy)

        print('    autoencoder_loss_val: ' + str(float(autoencoder_loss_val)) + ' - discriminator_loss_val: ' + str(float(discriminator_loss_val)))
        autoencoder_losses_val.append(autoencoder_loss_val)
        discriminator_losses_val.append(discriminator_loss_val)
    
    return autoencoder_losses_train, discriminator_losses_train, autoencoder_losses_val, discriminator_losses_val


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
def plot_image_grid(image_arr, save_grid_filename = None):
    plt.clf()

    result_figsize_resolution = 40 # 1 = 100px

    images_count = image_arr.shape[0]
    print('Images count: ', images_count)

    # Calculate the grid size:
    grid_size = math.ceil(math.sqrt(images_count))

    # Create plt plot:
    _, axes = plt.subplots(grid_size, grid_size, figsize = (result_figsize_resolution, result_figsize_resolution))

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
def plot_image(image, label_true = None, class_names = None, label_pred = None, save_img_filename = None):
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


''' Plot given training data (wrapped as a list of 1D array-likes) and save resulting plot to file with given name '''
def plot_training_data(data, title, ylabel, legend_labels, filename):
    
    # Clear plot
    plt.clf()

    # Plot every array of data (given as 1D array-likes) and find min and max value out of all arrays
    min_data_val = 9999999
    max_data_val = -9999999
    for d in data:

        # Plot array of data
        plt.plot(d)

        # Find min and max value out of all arrays
        d_np = np.asarray(d)
        cur_min = np.min(d_np)
        if(cur_min < min_data_val):
            min_data_val = cur_min
        cur_max = np.max(d_np)
        if(cur_max > max_data_val):
            max_data_val = cur_max


    # Ensure legend does not cover the plots
    y_range = max_data_val - min_data_val
    plt.ylim([min_data_val - 0.05*y_range, max_data_val + 0.45*y_range])

    # Label the plot
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('Epoch')
    plt.legend(legend_labels, loc = 'upper right')

    # Save plot as file with given file name
    plt.savefig(filename)


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
            architecture = sys.argv[cmd_arg_it].lower()
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

    # Verify we got at least one attacked image
    if(len(X_attacked) < 1):
        print('Could not load attacked images')
        exit()

    # Create and train or load in existing model
    if(architecture != 'blur'):
        if should_train:
            if(architecture == 'fully_conv'):
                model = create_fully_conv_denoising_autoencoder(X_attacked[0].shape)
                num_epochs = 1000

            elif(architecture == 'fully_conn'):
                model = create_fully_conn_denoising_autoencoder(X_attacked[0].shape)
                num_epochs = 10000

            elif(architecture == 'gan'):
                autoencoder, discriminator = create_GAN_denoising_autoencoder(X_attacked[0].shape)
                num_epochs = 1000
                autoencoder_losses_train, discriminator_losses_train, autoencoder_losses_val, discriminator_losses_val = train_GAN_denoising_autoencoder(X_attacked_train, X_unattacked_train, X_attacked_test, X_unattacked_test, num_epochs, 256, autoencoder, discriminator)
                plot_training_data([autoencoder_losses_train, discriminator_losses_train, autoencoder_losses_val, discriminator_losses_val], 'GAN Denoising Autoencoder Loss During Training', 'Loss', ['autoencoder train', 'discriminator train', 'autoencoder test', 'discriminator test'], 'GAN_loss_vs_epochs.png')
                visualkeras.layered_view(discriminator, to_file = 'gan_discriminator_viz.png', legend = True, scale_xy = 1, scale_z = 1)
                model = autoencoder
            
            if(architecture != 'gan'):
                history = model.fit(X_attacked_train, X_unattacked_train, epochs = num_epochs, batch_size = 256, shuffle = True, validation_data = (X_attacked_test, X_unattacked_test))
                plot_training_data([history.history['loss'], history.history['val_loss']], architecture + ' Denoising Autoencoder Loss During Training', 'Loss', ['train', 'test'], architecture + '_loss_vs_epochs.png')
            
            model.save('./trained_' + architecture)
        else:
            if not os.path.exists('./trained_' + architecture):
                print('Existing model for architecture ' + architecture + ' not found; must be trained using --train')
                exit()
            model = models.load_model('./trained_' + architecture)

        # Get images denoised by given NN architecture
        denoised_attacked_images = model.predict(X_attacked_test)
        denoised_unattacked_images = model.predict(X_unattacked_test)

        # Create image visualizing model architecture
        visualkeras.layered_view(model, to_file = architecture + '_model_viz.png', legend = True, scale_xy = 1, scale_z = 1)
    else:

        # Get images that are slightly Gaussian blurred for non-NN baseline
        denoised_attacked_images = []
        for attacked_image in X_attacked_test:
            denoised_attacked_images.append(gaussian_filter(attacked_image, sigma = 0.75))

        denoised_unattacked_images = []
        for unattacked_image in X_unattacked_test:
            denoised_unattacked_images.append(gaussian_filter(unattacked_image, sigma = 0.75))
        
        denoised_attacked_images = np.asarray(denoised_attacked_images)
        denoised_unattacked_images = np.asarray(denoised_unattacked_images)

    # Save a grid of test images for each of the following image types: attacked, unattacked, and denoised
    plot_image_grid(X_attacked_test[:9] * 255, save_grid_filename = 'attacked_imgs.png')
    plot_image_grid(X_unattacked_test[:9] * 255, save_grid_filename = 'unattacked_imgs.png')
    plot_image_grid(denoised_attacked_images[:9] * 255, save_grid_filename = 'denoised_attacked_imgs_' + architecture + '.png')
    plot_image_grid(denoised_unattacked_images[:9] * 255, save_grid_filename = 'denoised_unattacked_imgs_' + architecture + '.png')