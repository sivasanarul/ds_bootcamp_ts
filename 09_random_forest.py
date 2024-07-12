# Import necessary libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Load the Iris dataset as an example
iris = datasets.load_iris()
X = iris.data  # Features
y = iris.target  # Target variable

# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
# Create a Random Forest classifier with 100 trees
clf_100 = RandomForestClassifier(n_estimators=100, criterion='log_loss', min_samples_leaf=4, random_state=42)
clf_100.fit(X_train, y_train)
y_pred_100 = clf_100.predict(X_test)
accuracy_100 = accuracy_score(y_test, y_pred_100)
print("Accuracy with 100 trees: {:.2f}%".format(accuracy_100 * 100))

# Create a Random Forest classifier with 50 trees
clf_50 = RandomForestClassifier(n_estimators=50, criterion='log_loss', min_samples_leaf=4, random_state=42)
clf_50.fit(X_train, y_train)
y_pred_50 = clf_50.predict(X_test)
accuracy_50 = accuracy_score(y_test, y_pred_50)
print("Accuracy with 50 trees: {:.2f}%".format(accuracy_50 * 100))

# Plot feature importances
features = iris.feature_names
importances_100 = clf_100.feature_importances_
importances_50 = clf_50.feature_importances_

x = np.arange(len(features))

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.bar(x, importances_100, align='center', color='blue')
plt.xticks(x, features, rotation=45)
plt.title('Feature Importances (100 Trees)')
plt.xlabel('Feature')
plt.ylabel('Importance')

plt.subplot(1, 2, 2)
plt.bar(x, importances_50, align='center', color='green')
plt.xticks(x, features, rotation=45)
plt.title('Feature Importances (50 Trees)')
plt.xlabel('Feature')
plt.ylabel('Importance')

plt.tight_layout()
plt.show()