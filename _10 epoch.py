#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout 
from tensorflow.keras.optimizersimport Adam 
from tensorflow.keras.callbacks import History import timeit


# In[11]:


# Load the dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the data
x_train = x_train.reshape((60000, 28, 28, 1)).astype('float32') / 255
x_test = x_test.reshape((10000, 28, 28, 1)).astype('float32') / 255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# In[12]:


# Defining the Multilayer Perceptron model
mlp_model = Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
mlp_model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=['accuracy'])


# In[13]:


# Defining the CNN model
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

cnn_model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=['accuracy'])


# In[14]:


# Define the LeNet model
lenet_model = Sequential([
    Conv2D(6, (5, 5), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(16, (5, 5), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(120, activation='relu'),
    Dense(84, activation='relu'),
    Dense(10, activation='softmax')
])

lenet_model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=['accuracy'])


# In[15]:


# Define LeNet-5 model
lenet5_model = Sequential([
    Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(16, kernel_size=(5, 5), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(120, activation='relu'),
    Dense(84, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
lenet5_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[16]:


from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Add
from tensorflow.keras.models import Model

# Define ResNet model
def resnet_model(input_shape, num_classes):

    inputs = Input(shape=input_shape)
    # Convolutional Block 1
    x = Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu')(inputs)
    x = Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Residual Block 1
    residual_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(2,2), padding='same')(x)
    residual_1 = BatchNormalization()(residual_1)
    
    x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    # Use a 1x1 convolution to match the shape of the skip connection
    skip_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(2,2), padding='same')(x)
    skip_1 = BatchNormalization()(skip_1)

    # Make sure shapes match before adding
    x = Add()([skip_1, residual_1])
    
    # Convolutional Block 2
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Residual Block 2
    residual_2 = Conv2D(filters=128, kernel_size=(1, 1), strides=(2,2), padding='same')(x)
    residual_2 = BatchNormalization()(residual_2)
    
    x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    # Use a 1x1 convolution to match the shape of the skip connection
    skip_2 = Conv2D(filters=128, kernel_size=(1, 1), strides=(2,2), padding='same')(x)
    skip_2 = BatchNormalization()(skip_2)

    # Make sure shapes match before adding
    x = Add()([skip_2, residual_2])
    
    # Global Average Pooling
    x = GlobalAveragePooling2D()(x)
    
    # Fully Connected Layers
    x = Dense(units=256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(units=128, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    outputs = Dense(units=num_classes, activation='softmax')(x)
    
    # Create model
    model = Model(inputs, outputs, name='ResNet')
    
    return model


# parameters
input_shape = (28, 28, 1)
num_classes = 10

# Instantiate ResNet model
resnet = resnet_model(input_shape, num_classes)

# Compile the model
resnet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])





# In[19]:


#traning the mlp model
import time

start_time = time.time()
mlp_history = mlp_model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test))
end_time = time.time()

print("Time taken: {:.2f} seconds".format(end_time - start_time))


# In[20]:


# Train the CNN Model

start_time = time.time()
cnn_history = cnn_model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test))
end_time = time.time()

print("Time taken: {:.2f} seconds".format(end_time - start_time))


# In[24]:


# Training LeNet Model
start_time= time.time()
lenet_history = lenet_model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test))
end_time = time.time()

print("Time taken: {:.2f} seconds".format(end_time - start_time))


# In[25]:


# Train the LetNet5 model
start_time= time.time()
lenet5_history = lenet5_model.fit(x_train, y_train, batch_size=128, epochs=10,validation_data=(x_test, y_test))
end_time = time.time()
print("Time taken: {:.2f} seconds".format(end_time - start_time))


# In[26]:


# ResNet training
start_time= time.time()
resnet_history = resnet.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test))
end_time = time.time()
print("Time taken: {:.2f} seconds".format(end_time - start_time))


# In[28]:


# Plot the loss for each model
plt.plot(mlp_history.history['loss'], label='MLP')
plt.plot(cnn_history.history['loss'], label='CNN')
plt.plot(lenet_history.history['loss'], label='LeNet')
plt.plot(lenet5_history.history['loss'], label='LeNet-5')
plt.plot(resnet_history.history['loss'], label='ResNet')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[30]:


# Plot the accuracy for each model
plt.plot(mlp_history.history['accuracy'], label='MLP')
plt.plot(cnn_history.history['accuracy'], label='CNN')
plt.plot(lenet_history.history['accuracy'], label='LeNet')
plt.plot(lenet5_history.history['accuracy'], label='LeNet-5')
plt.plot(resnet_history.history['accuracy'], label='ResNet')
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[ ]:




