import numpy as np
from read_data import get_train_data, get_test_data, visualize_points
from keras.models import Sequential
from keras.layers import (
    Conv2D, MaxPooling2D, Flatten,
    Dense, Dropout
)
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from skimage.transform import resize
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import os  # Import os module for directory creation

# Function to create directory for weights if it does not exist
def create_weights_directory():
    os.makedirs('weights', exist_ok=True)

# Get the preprocessed train and test data
imgs_train, points_train = get_train_data()
imgs_test = get_test_data()


# Define the architecture
def get_model():
    model = Sequential()
    model.add(Conv2D(64, kernel_size=3, strides=2, padding='same', input_shape=(96,96,1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Conv2D(64, kernel_size=1, strides=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Flatten())
    
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(30))
    return model

def compile_model(model):       # Compile the model
    model.compile(loss='mean_squared_error', optimizer=Adam(), metrics=['accuracy'])

def train_model(model):         # Fit the model
    checkpoint = ModelCheckpoint(filepath='weights/checkpoint-{epoch:02d}.hdf5')
    model.fit(imgs_train, points_train, epochs=300, batch_size=100, callbacks=[checkpoint])

# Load weights for a previously trained model
def load_trained_model(model):
    model.load_weights('weights/checkpoint-300.hdf5')

# Testing the model
def test_model(model):    
    test_img = imgs_test[0]  # Test on the first sample of the test set
    test_img_input = np.reshape(test_img, (1,96,96,1))      # Model takes input of shape = [batch_size, height, width, no. of channels]
    prediction = model.predict(test_img_input)      # shape = [batch_size, values]
    visualize_points(test_img, prediction[0])
    
    # Test on first 10 samples of the test set
    for i in range(1, 11):
        test_img = imgs_test[i]
        test_img_input = np.reshape(test_img, (1,96,96,1))      # Model takes input of shape = [batch_size, height, width, no. of channels]
        prediction = model.predict(test_img_input)      # shape = [batch_size, values]
        visualize_points(test_img, prediction[0])

# Create weights directory before training
create_weights_directory()

# Train the model
model = get_model()
compile_model(model)
#train_model(model)  # Commented out to prevent training during testing
load_trained_model(model)
test_model(model)
