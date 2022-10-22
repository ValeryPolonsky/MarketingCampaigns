# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

# Importing the dataset
dataset_train = pd.read_csv('data_classifiers_training.csv')
dataset_test = pd.read_csv('data_classifiers_test.csv')

# Splitting the dataset into the Training set and Test set
X_train = dataset_train.iloc[:, :-1].values
y_train = dataset_train.iloc[:, -1].values
X_test = dataset_test.iloc[:, :-1].values
y_test = dataset_test.iloc[:, -1].values

# Training the Decision Tree Classification model on the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'gini', 
                                    splitter='best',
                                    min_samples_split = 2,                                  
                                    random_state = 0)
classifier.fit(X_train, y_train)

# Initializing & training the ANN
classifier = tf.keras.models.Sequential()
classifier.add(tf.keras.layers.Dense(units=4, activation='relu'))
classifier.add(tf.keras.layers.Dense(units=4, activation='relu'))
classifier.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
classifier.compile(optimizer = 'adam', 
                    loss = 'binary_crossentropy', 
                    metrics = ['accuracy'])
classifier.fit(X_train, y_train, batch_size = 81, epochs = 100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
acc_score = accuracy_score(y_test, y_pred)

# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'criterion': ['gini'], 
               'splitter': ['best'],
               'min_samples_split': [3],
               'min_samples_leaf': [1],
               'min_weight_fraction_leaf': [0],
               'min_impurity_decrease': [0],
               'ccp_alpha': [0,]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))
