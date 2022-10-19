# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

# Importing the dataset
sample_size_yes = 1000
sample_size_no = 1000

dataset = pd.read_csv('data.csv')
dataset_yes = dataset[dataset["deposit"] == 'yes'].sample(n = sample_size_yes)
dataset_no = dataset[dataset["deposit"] == 'no'].sample(n = sample_size_no)
dataset_merged = pd.concat([dataset_yes,dataset_no])
dataset_merged = dataset_merged.sample(frac=1).reset_index(drop=True)

X = dataset_merged.iloc[:, :-1].values
y = dataset_merged.iloc[:, -1].values

# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, [0,5,9,11,12,13,14]])
X[:, [0,5,9,11,12,13,14]] = imputer.transform(X[:, [0,5,9,11,12,13,14]])

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer.fit(X[:, [1,2,3,4,6,7,8,10,15]])
X[:, [1,2,3,4,6,7,8,10,15]] = imputer.transform(X[:, [1,2,3,4,6,7,8,10,15]])

# Encoding variables
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:,1] = le.fit_transform(X[:,1])
X[:,2] = le.fit_transform(X[:,2])
X[:,3] = le.fit_transform(X[:,3])
X[:,4] = le.fit_transform(X[:,4])
X[:,6] = le.fit_transform(X[:,6])
X[:,7] = le.fit_transform(X[:,7])
X[:,8] = le.fit_transform(X[:,8])
X[:,10] = le.fit_transform(X[:,10])
X[:,15] = le.fit_transform(X[:,15])
y = le.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

print('Y_train YES: {0}'.format(len(y_train[y_train == 1])))
print('Y_train NO: {0}'.format(len(y_train[y_train == 0])))
print('Y_test YES: {0}'.format(len(y_test[y_test == 1])))
print('Y_test NO: {0}'.format(len(y_test[y_test == 0])))

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Initializing & training the ANN
classifier = tf.keras.models.Sequential()
classifier.add(tf.keras.layers.Dense(units=10, activation='relu'))
classifier.add(tf.keras.layers.Dense(units=10, activation='relu'))
classifier.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
classifier.compile(optimizer = 'adam', 
                   loss = 'binary_crossentropy', 
                   metrics = ['accuracy'])
classifier.fit(X_train, y_train, batch_size = 33, epochs = 200)

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
