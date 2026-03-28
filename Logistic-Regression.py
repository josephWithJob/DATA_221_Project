# Author: Lucas Joffre
# Date: March 27 2026
# DATA221 Final Project - Logistic-Regression

# Library Imports
import pandas
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Reads the csv
airline_data_csv = pandas.read_csv("AirlineData.csv")

# Separates the data for the target data
satisfaction_airline_data_csv = airline_data_csv['satisfaction']

# Separates the feature data and removes unnecessary columns of id number and first column unnamed.
feature_data_from_airline_data_csv = airline_data_csv.drop('satisfaction', axis=1)
feature_data_from_airline_data_csv = feature_data_from_airline_data_csv.drop('id', axis=1)
feature_data_from_airline_data_csv = feature_data_from_airline_data_csv.iloc[:, 1:]

# Turns classification data into 0's and 1's
satisfaction_airline_data_csv = satisfaction_airline_data_csv.replace({"satisfied":1, "neutral or dissatisfied":0})

feature_data_from_airline_data_csv = feature_data_from_airline_data_csv.replace({"Male":0, "Female":1, "disloyal Customer":0, "Loyal Customer":1,
                                                                                 "Personal Travel":0, "Business travel":1, "Eco":0, "Eco Plus":0.5,
                                                                                 "Business":1})
# force the satisfaction labels to be integers 0 and 1
# any value that didn't get replaced by 0 or 1 will be caught here
satisfaction_airline_data_csv = satisfaction_airline_data_csv.astype(int)

# remove any row that has at least one missing value
feature_data_from_airline_data_csv = feature_data_from_airline_data_csv.dropna()


# this removes the rows in our target columns that we've removed from the feature columns
satisfaction_airline_data_csv = satisfaction_airline_data_csv[feature_data_from_airline_data_csv.index]

# Splits data for training ML models
features_train, features_test, labels_train, labels_test = train_test_split(feature_data_from_airline_data_csv,
                                                                            satisfaction_airline_data_csv, test_size=0.20, random_state=42)


classifier = LogisticRegression(max_iter=1000)
classifier.fit(features_train, labels_train)

accuracy_score = classifier.score(features_test, labels_test)
print(f"Model Accuracy: {accuracy_score}")