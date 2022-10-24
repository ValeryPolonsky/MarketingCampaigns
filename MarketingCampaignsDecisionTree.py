# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

# Training the Decision Tree Classification model on the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'gini', 
                                    splitter='best',
                                    min_samples_split = 3,                                  
                                    random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

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
