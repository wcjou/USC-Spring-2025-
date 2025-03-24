import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.neural_network import MLPRegressor
import urllib.request

# data source: https://www.tensorflow.org/tutorials/keras/regression

# use this command to download the file directly to the computer: curl -o output.file.name.here 'url-here'
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

raw_dataset = pd.read_csv(url, names=column_names,
                          na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True)

# save url directly to csv
urllib.request.urlretrieve(url, 'cars_raw.csv')
# create a csv from the Dataframe
raw_dataset.to_csv('cars.csv', index=False)

dataset = raw_dataset.copy()
dataset.tail()
dataset.isna().sum()
dataset = dataset.dropna()

# dummys
dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')
dataset.tail()

sns.pairplot(dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')
plt.show()

print(dataset)

x = dataset.iloc[:,1:]
y = dataset.iloc[:,0]

# Standardize all variables other than Outcome.
standard = StandardScaler()
standard.fit(x)
x = pd.DataFrame(standard.transform(x), columns=x.columns)

# X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=2023)

# Create a simple instance of a keras sequential NN model
model1 = tf.keras.Sequential()
model1.add(tf.keras.Input(shape=(1,)))
model1.add(tf.keras.layers.Dense(units=1))

# Create a complicated instance of a keras sequential NN model
model2 = tf.keras.Sequential()
model2.add(tf.keras.Input(shape=(1,)))
model2.add(tf.keras.layers.Dense(64, activation='relu')),
model2.add(tf.keras.layers.Dense(64, activation='relu')),
model2.add(tf.keras.layers.Dense(units=1))

# Set parameters for loss function, optimizer and metrics
model1.compile(optimizer="sgd", loss="mean_squared_error")
model2.compile(optimizer="sgd", loss="mean_squared_error")

# Let's compare with MLPRegressor
model3 = MLPRegressor(hidden_layer_sizes=(64,64), activation="relu",
                    max_iter=100, alpha=0.01, solver="sgd",
                    random_state=2022, learning_rate_init=0.01, learning_rate='adaptive', verbose=False)

# We now train the network on a specific part of the data because we want to demonstrate a 1-parameter regression
# To enable us to change the variable in only 1 place we generically call it variable
variable = x.iloc[:,2]
# Be aware: Keras needs a series as input
# Reminder: MLPRegressor needs a dataset of shape (:,1) as input and not (:,)
variable_extra_dim = variable.values.reshape(-1,1)

# We fit the models; Keras can do a train/validation test data split, MLP cannot do
# that and this would have to be implemented manually using a classic training/test data split
history1 = model1.fit(variable, y, validation_split=0.33, epochs=100)
history2 = model2.fit(variable, y, validation_split=0.33, epochs=100)
history3 = model3.fit(variable_extra_dim, y)

# summarize the model set up
print(model1.summary())
print(model2.summary())

# display relationship between and y
print(model1.get_weights())
print(model2.get_weights())

# Plotting the Loss Curves
# plt.plot(history1.history['loss'])
# plt.plot(history1.history['val_loss'])
plt.plot(history2.history['loss'])
plt.plot(history2.history['val_loss'])
plt.plot(model3.loss_curve_)
plt.title('model loss')
plt.ylabel('accuracy')
plt.xlabel('epoch')
# plt.legend(['Simple Keras', 'Simple Keras test', 'Complex Keras', 'Complex Keras test', 'Complex MLP'], loc='upper right')
plt.legend(['Complex Keras', 'Complex Keras test', 'Complex MLP'], loc='upper right')
plt.show()

# Plotting the results
var_min = variable.min(axis=0)
var_max = variable.max(axis=0)

x_plot = np.linspace(var_min, var_max, num=50)
# again, for MLPRegressor we need input of shape (:,1) and not (:,)
x_plot_extra_dim = x_plot[:,np.newaxis]
y_pred1 = model1.predict(x_plot)
y_pred2 = model2.predict(x_plot)
y_pred3 = model3.predict(x_plot_extra_dim)

# Plot x, y as a scatter plot
# on the same figure, plot x_plot and y_pred as a line plot
plt.scatter(variable, y)
plt.plot(x_plot, y_pred1, color='red', linestyle='dashed',linewidth=2, label="Simple Keras model with "+ str(model1.count_params()) +" weights")
plt.plot(x_plot, y_pred2, color='green', linestyle='dashed',linewidth=2, label="Complex Keras model with "+ str(model2.count_params()) +" weights")
plt.plot(x_plot, y_pred3, color='purple', linestyle='dashed',linewidth=2, label="Complex model with MLPRegressor")
plt.legend()
plt.xlabel(str(variable.name))
plt.ylabel(str(y.name))
plt.show()




