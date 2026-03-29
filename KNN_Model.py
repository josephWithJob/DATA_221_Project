# Harris Khan
# March 25, 2026
# DATA221 Final Project - KNN Model

# Data website link: https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction?resource=download

# Library Imports
import pandas
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier

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

# ========================= KNN MODEL =========================

# Define a list of 8 different number of neighbours to use to define our KNN Model
list_of_number_of_neighbors = [1,3,5,7,9,11,13, 15]

# Initialize evaluation metrics variables with the number of neighbours that produce the highest f1-score
highest_f1_score_of_knn_model = 0.0
accuracy_at_highest_f1_score = 0.0
precision_at_highest_f1_score = 0.0
recall_at_highest_f1_score = 0.0

number_of_neighbors_at_highest_f1_score = 0

for num_neighbors in list_of_number_of_neighbors:
    # Define a KNN model that searches the selected number of closest neighbours
    knn_model_of_customer_satisfaction = KNeighborsClassifier(n_neighbors=num_neighbors)

    # Train the KNN model using the training data
    trained_knn_model_of_customer_satisfaction = knn_model_of_customer_satisfaction.fit(features_train, labels_train)

    # Make predictions using the testing data
    predictions_of_knn_model = trained_knn_model_of_customer_satisfaction.predict(features_test)

    # Calculate evaluation metrics
    f1_score_of_knn_model = f1_score(labels_test, predictions_of_knn_model)
    accuracy_of_knn_model = accuracy_score(labels_test, predictions_of_knn_model)
    precision_of_knn_model = precision_score(labels_test, predictions_of_knn_model)
    recall_of_knn_model = recall_score(labels_test, predictions_of_knn_model)

    # Find the highest f1-score of all KNN Models and set their evaluation metric scores here
    if f1_score_of_knn_model > highest_f1_score_of_knn_model:
        highest_f1_score_of_knn_model = f1_score_of_knn_model
        accuracy_at_highest_f1_score = accuracy_of_knn_model
        precision_at_highest_f1_score = precision_of_knn_model
        recall_at_highest_f1_score = recall_of_knn_model

        number_of_neighbors_at_highest_f1_score = num_neighbors

# Print the evaluation metrics
print("Number of K-Nearest-Neighbors that Produce Highest F1-Score:", number_of_neighbors_at_highest_f1_score)
print("Highest F1-Score of the KNN model: ", highest_f1_score_of_knn_model)
print("Accuracy at the Highest F1-Score: ", accuracy_at_highest_f1_score)
print("Precision at the Highest F1-Score: ", precision_at_highest_f1_score)
print("Recall at the Highest F1-Score: ", recall_at_highest_f1_score)
