
#importing all the required libraries used for building the model
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

# Dataset directly available in the keras 
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

#creating the CNN model
classifier = Sequential()

classifier.add(Conv2D(32,3, input_shape=(28,28,1), padding='same', activation='relu'))
classifier.add(Conv2D(64,3, padding='same', activation='relu'))
classifier.add(MaxPooling2D(2,2))

#adding a second convolution layer to improve the performance (No of features has increased as the size of the image reduces)
classifier.add(Conv2D(128,3, padding='same', activation='relu'))
classifier.add(Conv2D(256,3, padding='same', activation='relu'))
classifier.add(MaxPooling2D(2,2,data_format='channels_last'))

classifier.add(Flatten())

#creating the ANN (part of the CNN) where the pixels are the input  
classifier.add(Dense(256, activation='relu'))
classifier.add(Dense(128, activation='relu'))

#using softmax as the ouptut has more than one nodes, so sigmoid won't give values that add up to one
classifier.add(Dense(10, activation='softmax'))


# Optimizer - what method are you usung to optimize the loss (gradient descent etc.)
# loss - what function are you considering to optimize (mean squared error, absolute difference)
# loss is like a metric but the loss values are considered while back propagating where as the 
# metrics values are not, metrics is used for other purposes such as call backs
classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# image preprocessing 

# code taken from the keras documentation, helps us to generate more images so that we can extract 
# better features, this is done batch wise
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# altering the various characteristics of the image
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

# changing the dimensions of the matrices so that they are compatible with the upcoming code
x_train = np.array(x_train)
x_train = x_train.reshape([60000,28,28,1])
y_train = y_train.reshape([60000,1])
print(x_train.shape)
print(y_train.shape)

# Fitting the image generator function to the training sample
datagen.fit(x_train)

x_test = np.array(x_test)
x_test = x_test.reshape([10000,28,28,1])
y_test = np.array(y_test)
y_test = y_test.reshape([10000,1])


# fits the model on batches with real-time data augmentation:
# steps per epoch = training set size
# batch_size = no of data rows it'll process before altering the weights in the neural network
remember = classifier.fit_generator(datagen.flow(x_train, y_train, batch_size=32), epochs=25, validation_data=(x_test, y_test))


print(x_test.shape)

# predicting the output of the test sample
y_pred = classifier.predict(x_test)

# getting the result from the predicted values (as we use softmax)
y_pred_ac = []
for p in range (10000):
    y_pred_ac.append(0)

for i in range(10000):
    ma = y_pred[i][0]
    for j in range(10):
        if (y_pred[i][j]>ma): 
            y_pred_ac[i]= j
            ma = y_pred[i][j]

y_pred_ac = np.array(y_pred_ac)

#creating the confusion matrix to evaluate our performance
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_ac)


# getting the accuracy of the predicted data of the test set
def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements 

accuracy(cm)
    
