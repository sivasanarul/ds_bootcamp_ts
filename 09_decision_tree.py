from sklearn.tree import DecisionTreeClassifier
import numpy as np
# Import necessary libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset as an example
iris = datasets.load_iris()
X = iris.data  # Features
y = iris.target  # Target variable
# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
print(len(X_train)

clf_gini = DecisionTreeClassifier(criterion='gini')
clf_gini.fit(X_train, y_train)
y_pred_gini = clf_gini.predict(X_test)
accuracy_gini = accuracy_score(y_test, y_pred_gini)
print("Gini criterion accuracy: {:.2f}%".format(accuracy_gini * 100))


print("I am adding this today")
##########################
# new_data_sample = np.array([[5.1, 3.5, 1.4, 0.2]])
#
# # Use the loaded model to predict the class of the new data sample
# prediction = clf_gini.predict(new_data_sample)
#
# # Output the prediction
# print(f"New data sample: {new_data_sample}")
# print(f"Predicted class: {prediction[0]}")
# print(f"Predicted class name: {iris.target_names[prediction][0]}")



# y_pred = [0, 2, 1, 3, 0]
# y_true = [0, 1, 2, 3, 0]
#
# We see matches on indices 0, 3 and 4. Thus:
#
# number of matches = 3
# number of samples = 5
#
# Finally, the accuracy calculation:
#
# accuracy = matches/samples
# accuracy = 3/5
accuracy = 0.6










# Using the Entropy criterion
clf_entropy = DecisionTreeClassifier(criterion='entropy')
clf_entropy.fit(X_train, y_train)
y_pred_entropy = clf_entropy.predict(X_test)
accuracy_entropy = accuracy_score(y_test, y_pred_entropy)
print("Entropy criterion accuracy: {:.2f}%".format(accuracy_entropy * 100))

# Using max depth 3, min samples split 4, and sqrt rule for features
max_features_sqrt = int(np.sqrt(X_train.shape[1]))
clf_custom = DecisionTreeClassifier(criterion='gini', max_depth=7, min_samples_split=4, max_features=max_features_sqrt)
clf_custom.fit(X_train, y_train)
y_pred_custom = clf_custom.predict(X_test)
accuracy_custom = accuracy_score(y_test, y_pred_custom)
print("Custom criterion accuracy: {:.2f}%".format(accuracy_custom * 100))
