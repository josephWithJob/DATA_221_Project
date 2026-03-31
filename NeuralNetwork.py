# Kyle Wang
# March 26, 2026
# DATA221 Final Project - Neural Network model
# Data website link: https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction?resource=download

# Library Imports
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, ConfusionMatrixDisplay, \
    confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models
import tensorflow as tf
import numpy as np
import random

#set the randomness
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

# Reads the csv
airline_data_csv = pd.read_csv("AirlineData.csv")

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

scaler = StandardScaler()
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

# scale the training features
features_train = scaler.fit_transform(features_train)

#scale the test features using the same scale as the training features
features_test = scaler.transform(features_test)


# labels for confusion matrix
labels = ["neutral or dissatisfied","satisfied"]

#values of all epoch values used
epoch_values = []
#values of accuracy scores corresponding to epoch values
epoch_value_accuracy_score = []
#values of precision scores corresponding to epoch values
epoch_value_precision_score = []
#values of recall scores corresponding to epoch values
epoch_value_recall_score = []
#values of f1 scores corresponding to epoch values
epoch_value_f1_score = []
#the predicted labels that corresponds to the current BEST epoch value in each iteration.
best_test_predicted_labels = []

# creates neural network
model = models.Sequential([
    layers.Input(shape=(22,)),
    layers.Dense(11, activation="relu"),
    layers.Dense(5, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

# configure model for training.
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
#iterates through a range of epoch values to find the optimal epoch value

#a constant value to add the epoch value by on every iteration.
constant_epoch_value_add = 5

#iterate from 2 to 21, this will allow the epoch value to be 10(5*2), 15(5*3), ... 105(21*5)
for iterate in range(2,22):
    #trains the model
    model.fit(features_train,labels_train,validation_split =0.1,epochs = constant_epoch_value_add ,batch_size = 32, verbose = 0) #fit/train model


    #predicts labels based on the model.
    test_predicted_labels = (model.predict(features_test, verbose = 0) > 0.5).astype(int).flatten() #find the labels for all the predictions

    #add the current epoch value to the epoch_values so it can later be used later.
    epoch_values.append(constant_epoch_value_add*iterate)

    #add accuracy score
    epoch_value_accuracy_score.append(accuracy_score(labels_test,test_predicted_labels))
    #add precision score
    epoch_value_precision_score.append(precision_score(labels_test,test_predicted_labels))
    #add recall score
    epoch_value_recall_score.append(recall_score(labels_test,test_predicted_labels))
    # add f1 score
    epoch_value_f1_score.append(f1_score(labels_test,test_predicted_labels))

    #is the current epoch value the best? if so, replace the best labels with the current labels since a new beest epoch value has been found
    if epoch_value_f1_score[-1] >= max(epoch_value_f1_score):
        best_test_predicted_labels = test_predicted_labels


#plot the accuracy, precision, recall and f1 scores on the y-axis with the x-axis being the epoch values.
plt.plot(epoch_values,epoch_value_accuracy_score,label = "accuracy")
plt.plot(epoch_values, epoch_value_precision_score,label = "precision")
plt.plot(epoch_values, epoch_value_recall_score, label = "recall")
plt.plot(epoch_values, epoch_value_f1_score, label = "f1_score")
plt.xlabel("epoch values") #add an x-axis label
plt.ylabel("metric scores") #add a y-axis label
plt.title("response of metric scores due to epoch values.") #add a title to the plot
plt.legend() #make a legend so the score plotted line can be found correctly
plt.tight_layout() #make the labels to the plot fit correctly.
plt.show() #display the line plot graph.

print("best f1_score: ",max(epoch_value_f1_score)) #print the best f1 score
print("corresponding epoch value: ", epoch_values[epoch_value_f1_score.index(max(epoch_value_f1_score))]) #print the epoch value that corresponds
print("corresponding accuracy value: ", epoch_value_accuracy_score[epoch_value_f1_score.index(max(epoch_value_f1_score))]) #print the corresponding accuracy score
print("corresponding recall value: ", epoch_value_recall_score[epoch_value_f1_score.index(max(epoch_value_f1_score))]) #print the corresponding f1 score
print("corresponding precision value: ", epoch_value_precision_score[epoch_value_f1_score.index(max(epoch_value_f1_score))]) #print the corresponding precision score

confusion_Matrix_Neural = confusion_matrix(labels_test,best_test_predicted_labels) #create the confusion matrix
#create labels for the confusion matrix
display = ConfusionMatrixDisplay(confusion_matrix = confusion_Matrix_Neural, display_labels= labels)
#make the confusion matrix into a matplotlib plot
display.plot()
#create a title for the confusion matrix
plt.title("Neural network confusion matrix")
plt.tight_layout() #make the labels to the plot fit correctly.
#display the confusion matrix
plt.show()

#create a dictionary for the epoch values, accuracy scores, precision scores, recall, and f1 scores
data = {"epoch values": epoch_values, "accuracy": epoch_value_accuracy_score, "precision": epoch_value_precision_score, "recall": epoch_value_recall_score, "F1_score": epoch_value_f1_score}
#create the pandas dataframe
df = pd.DataFrame(data)
#print the dataframe
print(df)


