import os, re, time, json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
import tensorflow_datasets as tfds

EPOCHS = 1
BATCH_SIZE = 64
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def preprocess_image_input(input_images):
  input_images = input_images.astype('float32')
  output_ims = tf.keras.applications.resnet50.preprocess_input(input_images)
  return output_ims

def feature_extractor(inputs):
    feature_extractor = tf.keras.applications.resnet.ResNet50(input_shape=(224, 224, 3),
                                                              include_top=False,
                                                              weights='imagenet')(inputs)
    return feature_extractor

def classifier(inputs):
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1024, activation="relu")(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dense(10, activation="softmax", name="classification")(x)
    return x

def final_model(inputs):
    resize = tf.keras.layers.UpSampling2D(size=(7,7))(inputs)
    resnet_feature_extractor = feature_extractor(resize)
    classification_output = classifier(resnet_feature_extractor)
    return classification_output

def define_compile_model():
  inputs = tf.keras.layers.Input(shape=(32,32,3))
  classification_output = final_model(inputs) 
  model = tf.keras.Model(inputs=inputs, outputs = classification_output)
  model.compile(optimizer='SGD', 
                loss='sparse_categorical_crossentropy',
                metrics = ['accuracy']) 
  return model

if __name__ == '__main__':
    (training_images, training_labels) , (validation_images, validation_labels) = tf.keras.datasets.cifar10.load_data()
    train_X = preprocess_image_input(training_images)
    valid_X = preprocess_image_input(validation_images)
    #model = define_compile_model()
    model = tf.keras.models.load_model('./trainedModel')
    #model.summary()
    #for _ in range(4):
    #    history = model.fit(train_X, training_labels, epochs=EPOCHS, validation_data = (valid_X, validation_labels), batch_size=64)
    #    model.save('./trainedModel')
    loss, accuracy = model.evaluate(valid_X, validation_labels)
    print(accuracy)