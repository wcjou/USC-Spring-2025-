import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# create 100 random x values. and then y values which are linearly
# dependent on x plus random noise

x = np.random.rand(100) * 10
# print(x)
y = 2 * x - 1 + np.random.randn(100)
# print(y)

# visualize x vs y
plt.scatter(x, y)
plt.show()

# Create an instance of a keras sequential NN model
model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(1,)))
model.add(tf.keras.layers.Dense(units=1))

# Set parameters for loss function, optimizer and metrics
model.compile(optimizer="sgd", loss="mean_squared_error")

# Train the network
model.fit(x, y, epochs=100)

# summarize the model set up
print(model.summary())
print(model.get_weights())

# make predictions for unseen (or test) x
x_test = np.linspace(-1, 11, num=50)
print(x_test.shape)
y_pred = model.predict(x_test)

# Plot x, y as a scatter plot
# on the same figure, plot x_test and y_pred as a line plot
plt.scatter(x, y)
plt.plot(x_test, y_pred)
plt.show()

