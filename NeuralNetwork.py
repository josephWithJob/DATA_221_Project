# Kyle Wang
# March 26, 2026
# DATA221 Final Project - Neural Network model
# Data website link: https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction?resource=download

# Library Imports
import pandas
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

# Reads the csv
airline_data_csv = pandas.read_csv("AirlineData.csv")

# Removes all NA value rows.
airline_data_csv = airline_data_csv.dropna()

# Turns classification data into 0's and 1's
airline_data_csv = airline_data_csv.replace({
    "Male":0, "Female":1,
    "disloyal Customer":0, "Loyal Customer":1,
    "Personal Travel":0, "Business travel":1,
    "Eco":0, "Eco Plus":1, "Business":2,
    "satisfied":1, "neutral or dissatisfied":0})

# Separates the data for the target data
satisfaction_airline_data_csv = airline_data_csv['satisfaction']

# Separates the feature data and removes unnecessary columns of id number and first column unnamed.
feature_data_from_airline_data_csv = airline_data_csv.drop('satisfaction', axis=1)
feature_data_from_airline_data_csv = feature_data_from_airline_data_csv.drop('id', axis=1)
feature_data_from_airline_data_csv = feature_data_from_airline_data_csv.iloc[:, 1:]


# Splits data for training ML models
features_train, features_test, labels_train, labels_test = train_test_split(
    feature_data_from_airline_data_csv,
    satisfaction_airline_data_csv,
    test_size=0.2,
    random_state=42)
