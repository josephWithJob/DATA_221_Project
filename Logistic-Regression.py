# Author: Lucas Joffre
# Date: March 27 2026
# DATA221 Final Project - Logistic-Regression

# Library Imports
import pandas
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt



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

#================================== Model ========================================
'''
Since our data set has drastic values such as flight distance our model has to work
with a lot of big and different values. We can scale our data so our model has an easier time
gathering up the data and comparing.
'''
# create a scaler
scaler = StandardScaler()

# fit the scaler on training data and transform both sets
features_train_scaled = scaler.fit_transform(features_train)
features_test_scaled = scaler.transform(features_test)

# train the model on the SCALED data using our scaler import
classifier = LogisticRegression(max_iter=100)
classifier.fit(features_train_scaled, labels_train)
f1_score = f1_score(labels_test, classifier.predict(features_test_scaled))

accuracy_score = classifier.score(features_test_scaled, labels_test)
print(f"Logistic Regression Model Accuracy: {accuracy_score:.2f}")
print(f"Logistic Regression Model f1_Score: {f1_score:.2f}")

# classifier coefficient contains the weight for each feature
importance_df = pd.DataFrame({
    'Feature': feature_data_from_airline_data_csv.columns,
    'Importance': classifier.coef_[0]
})

# sort them so the most important features are at the top
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# plot the results
plt.figure(figsize=(12, 8))
colors = ['green' if x > 0 else 'red' for x in importance_df['Importance']]
plt.barh(importance_df['Feature'], importance_df['Importance'], color=colors)
plt.axvline(x=0, color='black', linestyle='-', linewidth=1) # Zero line for reference
plt.xlabel("Importance")
plt.title("What Drives Airline Satisfaction?")
plt.gca().invert_yaxis() # highest importance at the top
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()