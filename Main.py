# Data website link: https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction?resource=download

# Library Imports
import pandas
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Reads the csv
airline_data_csv = pandas.read_csv("AirlineData.csv")

# Drop any rows containing missing values
airline_data_csv = airline_data_csv.dropna()

# Separates the data for the target data
satisfaction_airline_data_csv = airline_data_csv['satisfaction']

# Separates the feature data and removes unnecessary columns of id number and first column unnamed.
feature_data_from_airline_data_csv = airline_data_csv.drop('satisfaction', axis=1)
feature_data_from_airline_data_csv = feature_data_from_airline_data_csv.drop('id', axis=1)
feature_data_from_airline_data_csv = feature_data_from_airline_data_csv.iloc[:, 1:]

# Turns classification data into 0's and 1's for labels vector
satisfaction_airline_data_csv = satisfaction_airline_data_csv.replace({
    "satisfied":1, "neutral or dissatisfied":0})

# Correct dtype by converting values in labels vector to integers
satisfaction_airline_data_csv = satisfaction_airline_data_csv.astype(int)

# Turns classification data into 0's and 1's for feature matrix
feature_data_from_airline_data_csv = feature_data_from_airline_data_csv.replace({
    "Male":0, "Female":1,
    "disloyal Customer":0, "Loyal Customer":1,
    "Personal Travel":0, "Business travel":1,
    "Eco":0, "Eco Plus":0.5, "Business":1})

# Splits data for training ML models using an 80/20 training/testing split
features_train, features_test, labels_train, labels_test = train_test_split(
    feature_data_from_airline_data_csv,
    satisfaction_airline_data_csv,
    test_size=0.20,
    random_state=42)

# ========================= DECISION TREE =========================
# Creates the decision tree model
decision_tree_classifier = DecisionTreeClassifier(criterion='entropy', min_impurity_decrease=0.01, max_depth=5)

# Trains the model
decision_tree_classifier.fit(features_train, labels_train)

# Predicts data using decision tree
predicted_labels_of_decision_tree = decision_tree_classifier.predict(features_test)

# Evaluates performance of decision tree
accuracy_of_decision_tree = accuracy_score(labels_test, predicted_labels_of_decision_tree)
precision_score_of_decision_tree = precision_score(labels_test, predicted_labels_of_decision_tree)
recall_score_of_decision_tree = recall_score(labels_test, predicted_labels_of_decision_tree)
f1_score_of_decision_tree = f1_score(labels_test, predicted_labels_of_decision_tree)

print("Accuracy:", accuracy_of_decision_tree)
print("Precision:", precision_score_of_decision_tree)
print("Recall Score:", recall_score_of_decision_tree)
print("f1 Score:", f1_score_of_decision_tree)

# Plotting the model
# import matplotlib.pyplot as plt
# from sklearn.tree import plot_tree
# plt.figure(figsize=(15, 10))
# plot_tree(decision_tree_classifier, filled=True)
# plt.show()

