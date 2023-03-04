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

#Setting input and output. Tensor flow is expecting a 4d array since the first dimension is the batch size (here we set it to one), and third dimension is channels
np.random.seed(59)
img=np.reshape(np.random.rand(1024), (1,16,16,4))
output=np.reshape(np.random.rand(96), (1,4,4,6))

#Set weights to desired values

# model.layers[0].set_weights([np.full((5, 5, 4, 3), 0.2), np.full(3, 0.2)]) #Shape of weight matrix is (w,h,input_channels,kernels)
# model.layers[1].set_weights([np.full((3, 3, 3, 2), 0.3), np.full(2, 0.3)]) #Shape of weight matrix is (w,h,input_channels,kernels)
rands_0 = 2 * np.random.rand(303) - 1
rands_1 = 2 * np.random.rand(56) - 1
rands_2 = 2 * np.random.rand(114) - 1
weights_0 = np.reshape(rands_0[:300], (5, 5, 4, 3))
weights_1 = np.reshape(rands_1[:54], (3, 3, 3, 2))
weights_2 = np.reshape(rands_2[:108], (3, 3, 2, 6))
model.layers[0].set_weights([weights_0, rands_0[-3:]]) #Shape of weight matrix is (w,h,input_channels,kernels)
model.layers[1].set_weights([weights_1, rands_1[-2:]]) #Shape of weight matrix is (w,h,input_channels,kernels)
model.layers[2].set_weights([weights_2, rands_2[-6:]]) #Shape of weight matrix is (w,h,input_channels,kernels)
# print(model.summary())

#print needed values.
np.set_printoptions(precision=5)
# print('model output before:')
# print(model.predict(img))
sgd = optimizers.SGD(learning_rate=100)
model.compile(loss='MSE', optimizer=sgd, metrics=['accuracy'])
history=model.train_on_batch(img,output)
print('model output after:')
print(model.predict(img))

print('1st convolutional layer, 1st kernel weights:')
print(model.layers[0].get_weights()[0])
print('1st convolutional layer, 1st kernel bias:')
print(model.layers[0].get_weights()[1])