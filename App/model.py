from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load Iris dataset
iris = load_iris()

# Initialize feature data (X) and target variables (Y)
X = iris.data
Y = iris.target

# Initialize a RandomForestClassifier object
rf_classifier = RandomForestClassifier(n_estimators=100)

# Fit the RandomForestClassifier model to the dataset
rf_classifier.fit(X, Y)

# Save the trained RandomForestClassifier model to a file using pickle
pickle.dump(rf_classifier, open("model.pickle", "wb"))

# Load the saved RandomForestClassifier model from the file
model = pickle.load(open("model.pickle", "rb"))

# Use the loaded model to make predictions
sample = [[9, 2, 4, 9]]  # Adjust the values to match the four features
prediction = model.predict(sample)
print(prediction)
