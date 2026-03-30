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

features_train = scaler.fit_transform(features_train)
features_test = scaler.transform(features_test)



labels = ["neutral or dissatisfied","satisfied"]
epoch_values = []
epoch_value_accuracy_score = []
epoch_value_precision_score = []
epoch_value_recall_score = []
epoch_value_f1_score = []
best_test_predicted_labels = []

for epoch_value in range(10,100,5):
    # creates neural network
    model = models.Sequential([
        layers.Input(shape=(22,)),
        layers.Dense(11, activation="relu"),
        layers.Dense(5, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])

    # configure model for training.
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    #trains the model
    model.fit(features_train,labels_train,validation_split =0.1,epochs = epoch_value ,batch_size = 32, verbose = 0) #fit/train model


    #predicts labels based on the model.
    test_predicted_labels = (model.predict(features_test, verbose = 0) > 0.5).astype(int).flatten() #find the labels for all the predictions
    epoch_values.append(epoch_value)
    epoch_value_accuracy_score.append(accuracy_score(labels_test,test_predicted_labels))
    epoch_value_precision_score.append(precision_score(labels_test,test_predicted_labels))
    epoch_value_recall_score.append(recall_score(labels_test,test_predicted_labels))
    epoch_value_f1_score.append(f1_score(labels_test,test_predicted_labels))
    if epoch_value_f1_score[-1] >= max(epoch_value_f1_score):
        best_test_predicted_labels = test_predicted_labels



plt.plot(epoch_values,epoch_value_accuracy_score,label = "accuracy")
plt.plot(epoch_values, epoch_value_precision_score,label = "precision")
plt.plot(epoch_values, epoch_value_recall_score, label = "recall")
plt.plot(epoch_values, epoch_value_f1_score, label = "f1_score")
plt.xlabel("epoch values")
plt.ylabel("metric scores")
plt.title("response of metric scores due to epoch values.")
plt.legend()
plt.show()

print("best f1_score: ",max(epoch_value_f1_score))
print("corresponding epoch value: ", epoch_values[epoch_value_f1_score.index(max(epoch_value_f1_score))])
print("corresponding accuracy value: ", epoch_value_accuracy_score[epoch_value_f1_score.index(max(epoch_value_f1_score))])
print("corresponding recall value: ", epoch_value_recall_score[epoch_value_f1_score.index(max(epoch_value_f1_score))])
print("corresponding precision value: ", epoch_value_precision_score[epoch_value_f1_score.index(max(epoch_value_f1_score))])

confusion_Matrix_Neural = confusion_matrix(labels_test,best_test_predicted_labels)
display = ConfusionMatrixDisplay(confusion_matrix = confusion_Matrix_Neural, display_labels= labels)
display.plot()
plt.title("Neural network confusion matrix")
plt.show()


data = {"epoch values": epoch_values, "accuracy": epoch_value_accuracy_score, "precision": epoch_value_precision_score, "recall": epoch_value_recall_score, "F1_score": epoch_value_f1_score}
df = pd.DataFrame(data)
print(df)