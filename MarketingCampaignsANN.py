# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import random

# Importing the dataset
dataset = pd.read_csv('data.csv')
    
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Taking care of missing data
from sklearn.impute import SimpleImputer
numeric_columns = [0,5,9,11,12,13,14]
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, numeric_columns])
X[:,numeric_columns] = imputer.transform(X[:,numeric_columns])

from sklearn.impute import SimpleImputer
string_columns = [1,2,3,4,6,7,8,10,15]
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer.fit(X[:,string_columns])
X[:,string_columns] = imputer.transform(X[:,string_columns])

# Encoding variables
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for column in string_columns:
    X[:,column] = le.fit_transform(X[:,column])
y = le.fit_transform(y)
    
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Initializing & training the ANN
classifier = tf.keras.models.Sequential()
classifier.add(tf.keras.layers.Dense(units=8, activation='relu'))
classifier.add(tf.keras.layers.Dense(units=8, activation='relu'))
classifier.add(tf.keras.layers.Dense(units=8, activation='relu'))
classifier.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
classifier.compile(optimizer = 'adam', 
                    loss = 'binary_crossentropy', 
                    metrics = ['accuracy'])
classifier.fit(X_train, y_train, batch_size = 81, epochs = 200)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
    
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
acc_score = accuracy_score(y_test, y_pred)



