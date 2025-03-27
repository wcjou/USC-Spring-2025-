# William Jou
# ITP 259, Spring 2025
# Homework 5

import random
import pandas as pd
import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split


df = pd.read_csv("USC-Spring-2025-\Spring 2025 Courses\ITP 259\Homework\HW5\A_Z Handwritten Data.csv")

print(df.head())
# target is the labels in the first column

X = df.iloc[:,1:]
y = df.iloc[:,0]

print(X.shape, y.shape)

# the target variables are numbers

# create a dictionary to map the numbers to the letters
letter_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G',
                7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M',
                  13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
                    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

# show a histogram count of the letters, and make the labels the letters
plt.figure(1)
ax = sb.countplot(x="label", data=df)
ax.set_xticklabels([letter_dict[i] for i in range(26)])
plt.show()

random_indices = np.random.choice(df.index, 64, replace=False)
fig, axes = plt.subplots(8, 8, figsize=(10, 10))

for i, ax in enumerate(axes.flat):
    # Extract the pixel values and reshape them to 28x28
    pixel = df.iloc[random_indices[i], 1:]
    pixel = np.array(pixel).reshape(28, 28)
    
    # Get the label and map it to a letter
    label = letter_dict[y.iloc[random_indices[i]]]
    
    # Display the image
    ax.imshow(pixel, cmap="gray")
    ax.set_title(label)  # Add letter as title
    ax.axis("off")  # Hide axes

# Ploting
plt.tight_layout()
plt.show()


# Split the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2023, stratify=y)

# Scale the train and test features

X_train_scaled = X_train/255
X_test_scaled = X_test/255

# create an MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(128, 64, 32, 16), activation="relu",
                    max_iter=15, alpha=1e-3, solver="adam",
                    random_state=2023, learning_rate_init=0.01, verbose=True)

mlp.fit(X_train_scaled, y_train)

# Plot loss curve
plt.plot(mlp.loss_curve_)
plt.show()

# Accuracy
print("The accuracy is", mlp.score(X_test_scaled, y_test))

# Confusion matrix
y_pred = mlp.predict(X_test_scaled)
y_pred_series = pd.Series(y_pred, index=X_test_scaled.index)

labels = [letter_dict[i] for i in range(26)]

cm = confusion_matrix(y_pred, y_test)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot()
plt.show()

# display the predicted letter of the first row in the test set using a plot
first_image = np.array(X_test.iloc[0]).reshape(28, 28)
plt.imshow(first_image, cmap="gray")
plt.title("The predicted letter is "+ str(letter_dict[y_pred[0]]) + ". The actual letter was " + str(letter_dict[y_test.iloc[0]]))
plt.show()

# Display a failed prediction
failed_df = X_test[y_pred != y_test]
failed_indices = failed_df.index.values
random_failed_index = random.choice(failed_indices)
failed_image = np.array(X_test.loc[random_failed_index]).reshape(28, 28)
plt.imshow(failed_image, cmap="gray")
plt.title("The predicted letter is "+ str(letter_dict[y_pred_series.loc[random_failed_index]]) + ". The actual letter was " + str(letter_dict[y_test.loc[random_failed_index]]))
plt.show()


