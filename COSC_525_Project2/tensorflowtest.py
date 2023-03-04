import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from parameters import generateExample2

#Create a feed forward network
model=Sequential()

# Add two convolutional layers
model.add(layers.Conv2D(3, 5, input_shape=(16,16,4), activation='sigmoid'))
model.add(layers.Conv2D(2, 3, activation='sigmoid'))
model.add(layers.Conv2D(6, 3, activation='sigmoid'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

#Set weights to desired values

# model.layers[0].set_weights([np.full((5, 5, 4, 3), 0.2), np.full(3, 0.2)]) #Shape of weight matrix is (w,h,input_channels,kernels)
# model.layers[1].set_weights([np.full((3, 3, 3, 2), 0.3), np.full(2, 0.3)]) #Shape of weight matrix is (w,h,input_channels,kernels)
weights_0 = np.reshape(np.linspace(-0.1, 0.199, 300), (5, 5, 4, 3))
weights_1 = np.reshape(np.linspace(-0.2, 0.33, 54), (3, 3, 3, 2))
weights_2 = np.reshape(np.linspace(-0.4, 0.67, 108), (3, 3, 2, 6))
model.layers[0].set_weights([weights_0, np.array([0.200, 0.201, 0.202])]) #Shape of weight matrix is (w,h,input_channels,kernels)
model.layers[1].set_weights([weights_1, np.array([0.34, 0.35])]) #Shape of weight matrix is (w,h,input_channels,kernels)
model.layers[2].set_weights([weights_2, np.array([0.68, 0.69, 0.70, 0.71, 0.72, 0.73])]) #Shape of weight matrix is (w,h,input_channels,kernels)
# print(model.summary())

#Setting input. Tensor flow is expecting a 4d array since the first dimension is the batch size (here we set it to one), and third dimension is channels
img=np.reshape(np.linspace(0, 0.2, 1024), (1,16,16,4))

output=np.reshape(np.linspace(0, 0.2, 96), (1,4,4,6))
# output=np.reshape(np.linspace(0, 0.2, 384), (1,8,8,6))

#print needed values.
np.set_printoptions(precision=5)
# print('model output before:')
# print(model.predict(img))
sgd = optimizers.SGD(learning_rate=100)
model.compile(loss='MSE', optimizer=sgd, metrics=['accuracy'])
history=model.train_on_batch(img,output)
print('model output after:')
print(model.predict(img))

# print('1st convolutional layer, 1st kernel weights:')
# print(model.layers[0].get_weights()[0])
# print('1st convolutional layer, 1st kernel bias:')
# print(model.layers[0].get_weights()[1])