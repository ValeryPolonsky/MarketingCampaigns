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
classifier_1 = tf.keras.models.Sequential()
classifier_1.add(tf.keras.layers.Dense(units=16, activation='relu'))
classifier_1.add(tf.keras.layers.Dense(units=16, activation='relu'))
classifier_1.add(tf.keras.layers.Dense(units=16, activation='relu'))
classifier_1.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
classifier_1.compile(optimizer = 'adam', 
                    loss = 'binary_crossentropy', 
                    metrics = ['accuracy'])
classifier_1.fit(X_train, y_train, batch_size = 81, epochs = 200)

# Training the Random Forest Classification model on the Training set
from sklearn.ensemble import RandomForestClassifier
classifier_2 = RandomForestClassifier(n_estimators=100,
                                    criterion='entropy',
                                    min_samples_split=3,
                                    min_samples_leaf=1,
                                    n_jobs=-1,
                                    random_state=8,
                                    class_weight='balanced_subsample')
classifier_2.fit(X_train, y_train)

# Predicting the Test set results
y_pred_1 = classifier_1.predict(X_test)
y_pred_1 = (y_pred_1 > 0.5)
y_pred_1 = y_pred_1.astype(int).flatten()   
y_pred_2 = classifier_2.predict(X_test)

y_pred = []
for row in range(0,len(y_test)):
    if (y_pred_1[row] == y_pred_2[row]):
        y_pred.append(y_pred_1[row])
    else:
        index = random.randint(0, 1)
        if (index == 0):
            y_pred.append(y_pred_1[row])
        else:
            y_pred.append(y_pred_2[row])
y_pred = np.array(y_pred)
            
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm_1 = confusion_matrix(y_test, y_pred_1)
acc_score_1 = accuracy_score(y_test, y_pred_1)

cm_2 = confusion_matrix(y_test, y_pred_2)
acc_score_2 = accuracy_score(y_test, y_pred_2)

cm = confusion_matrix(y_test, y_pred)
acc_score = accuracy_score(y_test, y_pred)

