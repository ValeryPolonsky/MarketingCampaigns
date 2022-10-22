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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the Random Forest Classification model on the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100,
                                    criterion='entropy',
                                    min_samples_split=3,
                                    min_samples_leaf=1,
                                    n_jobs=-1,
                                    random_state=8,
                                    class_weight='balanced_subsample')
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
acc_score = accuracy_score(y_test, y_pred)

# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'n_estimators':[100],
               'criterion': ['entropy'], 
               'min_samples_split': [3],
               'min_samples_leaf': [1],
               'n_jobs': [-1],
               'random_state': [8],
               'class_weight': ['balanced_subsample']}]
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

