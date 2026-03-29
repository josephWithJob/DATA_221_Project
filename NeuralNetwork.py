# Kyle Wang
# March 26, 2026
# DATA221 Final Project - Neural Network model
# Data website link: https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction?resource=download

# Library Imports
import pandas
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
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
airline_data_csv = airline_data_csv.astype(float)


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

#creates neural network
model = models.Sequential([
    layers.Input(shape = (22,)),
    layers.Dense(11,activation = "relu"),
    layers.Dense(5, activation = "relu"),
    layers.Dense(1, activation="sigmoid")
])

#configure model for training.
model.compile(optimizer = "adam", loss = "binary_crossentropy",metrics = ["accuracy"])

#trains the model
model.fit(features_train,labels_train,validation_split =0.1,epochs = 15 ,batch_size = 32) #fit/train model

#predicts labels based on the model.
test_predicted_labels = (model.predict(features_test) > 0.5).astype(int).flatten() #find the labels for all the predictions

print("Accuracy: ", accuracy_score(test_predicted_labels,labels_test))
print("Precision: ", precision_score(test_predicted_labels,labels_test))
print("Recall: ", recall_score(test_predicted_labels,labels_test))
print("F1 Score: ", f1_score(test_predicted_labels, labels_test))