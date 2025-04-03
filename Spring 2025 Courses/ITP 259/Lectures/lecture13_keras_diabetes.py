# -*- coding: utf-8 -*-
"""Lecture14_Keras_diabetes.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1IBcxpYc7cBqQxeayjuFat0kLGvaT-osW
"""

import pandas as pd
data = pd.read_csv('sample_data/diabetes.csv')
x = data.drop("Outcome", axis=1)
y = data["Outcome"]

print(data.head())

"""Step-1) Define Keras Model
Model in Keras always defines as a sequence of layers. It means that we initialize the sequence model and add the layers one after the other which is executed as the sequence of the list. Practically we have to try experimenting with the process of adding and removing the layers until we are happy with our architecture.

The thing which you need to take care of is the first layer has the right number of input features which is specified using the input_dim parameter. we can specify the number of neurons as the first argument in a layer. to define activation function use activation argument.

In this example, We will define a fully connected network with three layers. To define the fully connected layer use the Dense class of Keras.

The first layer has 12 neurons and activation function as relu
The second hidden layer has 8 neurons and activation function as relu
Finally, at the output layer, we use 1 unit and activation as sigmoid because it is a binary classification problem.
"""

from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
# observe the right dimension of input data
model.add(Dense(12, input_dim=8, activation="relu"))
model.add(Dense(12, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

"""Step-2) Compile The Keras Model
When we compile the Keras model, it uses the backend numerical libraries such as TensorFlow or Theano. Whatever backend you are using automatically chooses the best way to represent the network on your hardware such as CPU, GPU, or TPU.

When we are compiling the model we must specify some additional parameters to better evaluate the model and to find the best set of weights to map inputs to outputs.

Loss Function – one must specify the loss function to evaluate the set of weights on which model will be mapped. we will use cross-entropy as a loss function which is actually known as binary cross-entropy used for binary classification.
Optimizer – second is the optimizer to optimize the loss. we will use adam which is a popular version of gradient descent and gives the best result in most problems.
"""

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

"""Step-3) Start Training (Fit the Model)
After successful compilation of the model, we are ready to fit data to the model and start training the neural network. Along with providing data to model, we need to define a number of epochs and batch size over which training occurs.

Epoch – one single pass through all the rows in the training dataset
Batch size – number of samples considered by the model before updating the weights.
"""

model.fit(x,y, epochs=100, batch_size=10)

"""Step-4) Evaluate the Model
After training the model let’s know the performance of a neural network. Model is always evaluated on a test set, In this example for sake of simplicity we have trained on a complete dataset but while working on any project you basically split the data and train the network.
"""

loss , accuracy = model.evaluate(x, y)
print("Model loss: %.2f"% (loss))
print("Model accuracy: %.2f"% (accuracy*100))

"""Step-5) Making Predictions
predict the output of new data by simply using predict method. we have a binary classification problem statement so the output will simply be 0 or 1. However, model.predit produces a probability first! Scitkit learn rounds automatically.

Scikit-learn's predict() returns an array of shape (n_samples, ), whereas Keras' returns an array of shape (n_samples, 1). The two arrays are equivalent for your purposes, but the one from Keras is a bit more general, as it more easily extends to the multi-dimensional output case. To convert from the Keras output to Sklearn's, simply call y_pred.reshape(-1).
"""

import numpy as np
y_pred = model.predict(x)
print(y_pred)
print([round(x[0]) for x in predictions])
y_pred_bin = [round(x[0]) for x in predictions]

import matplotlib.pyplot as plt
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y, y_pred_bin)

print(cnf_matrix)

print("Accuracy: ", metrics.accuracy_score(y, y_pred_bin))

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
ConfusionMatrixDisplay(confusion_matrix=cnf_matrix).plot()
plt.show()