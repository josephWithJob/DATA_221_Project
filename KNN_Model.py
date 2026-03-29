# Harris Khan
# March 25, 2026
# DATA221 Final Project - KNN Model

# Data website link: https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction?resource=download

# Library Imports
import pandas
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import ConfusionMatrixDisplay

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

# Define a dictionary to later be used to create a table of evaluation metrics:
dictionary_of_knn_evaluation_metrics = {"K-Nearest Neighbours": list_of_number_of_neighbors,
                                        "F1-Score": [],
                                        "Accuracy": [],
                                        "Precision": [],
                                        "Recall": []}

# Initialize evaluation metrics variables with the number of neighbours that produce the highest f1-score
highest_f1_score_of_knn_model = 0.0

labels_test_of_highest_f1_score = 0
predictions_of_highest_f1_score = 0




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

    # Append evaluation metrics to dictionary for table
    dictionary_of_knn_evaluation_metrics["F1-Score"].append(f1_score_of_knn_model)
    dictionary_of_knn_evaluation_metrics["Accuracy"].append(accuracy_of_knn_model)
    dictionary_of_knn_evaluation_metrics["Precision"].append(precision_of_knn_model)
    dictionary_of_knn_evaluation_metrics["Recall"].append(recall_of_knn_model)

    # Find the highest f1-score of all KNN Models and store the labels_test and label predictions used to find it
    if f1_score_of_knn_model > highest_f1_score_of_knn_model:
        labels_test_of_highest_f1_score = labels_test
        predictions_of_highest_f1_score = predictions_of_knn_model


# ========================= TABLES AND FIGURES =========================

# TABLE OF EACH KNN MODEL'S PERFORMANCE METRICS

# Create and print a table of KNN Evaluation Metrics
table_of_knn_evaluation_metrics = pandas.DataFrame(dictionary_of_knn_evaluation_metrics)
print("\nTable of KNN Model's Evaluation Metrics:\n", table_of_knn_evaluation_metrics)

# LINE GRAPH OF EACH KNN MODEL's PERFORMANCE METRICS

# Create a line graph by plotting each evaluation metric at each k-value
plt.plot(table_of_knn_evaluation_metrics["K-Nearest Neighbours"], table_of_knn_evaluation_metrics["F1-Score"], label="F1 Score")
plt.plot(table_of_knn_evaluation_metrics["K-Nearest Neighbours"], table_of_knn_evaluation_metrics["Accuracy"], label="Accuracy")
plt.plot(table_of_knn_evaluation_metrics["K-Nearest Neighbours"], table_of_knn_evaluation_metrics["Precision"], label="Precision")
plt.plot(table_of_knn_evaluation_metrics["K-Nearest Neighbours"], table_of_knn_evaluation_metrics["Recall"], label="Recall")

# Add labels and title to graph
plt.xlabel("Number of Neighbours (k)")
plt.ylabel("Score")
plt.title("KNN Performance Metrics vs Number of Neighbours")

# Calculate the row with the highest f1-score
best_row = table_of_knn_evaluation_metrics.loc[table_of_knn_evaluation_metrics["F1-Score"].idxmax()]

# Store the best k-value and best f1-score here
best_k = best_row["K-Nearest Neighbours"]
best_f1 = best_row["F1-Score"]

# Plot the vertical line showing the k-value with the best f1-score
plt.axvline(x=best_k, linestyle="--", label="Best k")

# Plot the legend, a grid layout, and show the graph
plt.legend()
plt.grid()
plt.show()

# CONFUSION MATRIX OF THE BEST KNN MODEL

# Create a confusion matrix using the labels_test and label predictions of the model that produces the highest f1-score
ConfusionMatrixDisplay.from_predictions(labels_test_of_highest_f1_score, predictions_of_highest_f1_score)

# Set the title of the confusion matrix and show it
plt.title("Confusion Matrix of Best KNN Model")
plt.show()