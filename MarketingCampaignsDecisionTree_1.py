# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
all_columns_without_deposit = ['age','job','marital','education','default',
                               'balance','housing','loan','contact','day',
                               'month','duration','campaign','pdays','previous',
                               'poutcome','cluster']
all_columns_without_cluster = ['age','job','marital','education','default',
                               'balance','housing','loan','contact','day',
                               'month','duration','campaign','pdays','previous',
                               'poutcome','deposit']
dataset = pd.read_csv('data_clusters.csv')
dataset_0 = dataset[dataset['deposit'] == 0]
dataset_0_clusters = dataset_0['cluster'].unique()
dataset_0_clusters.sort()
dataset_1 = dataset[dataset['deposit'] == 1]
dataset_1_clusters = dataset_1['cluster'].unique()
dataset_1_clusters.sort()

dataset = dataset[dataset['cluster'] == 0]
dataset = dataset[all_columns_without_cluster]
dataset = dataset.sample(frac=1).reset_index(drop=True)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Encoding the Dependent Variable
# =============================================================================
# from sklearn.preprocessing import OneHotEncoder
# enc = OneHotEncoder()
# y = enc.fit_transform(y.reshape(-1,1)).toarray()
# =============================================================================

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
                                    min_samples_split = 2,                                  
                                    random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# =============================================================================
# # Making the Confusion Matrix
# from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
# cm = multilabel_confusion_matrix(y_test, y_pred)
# acc_score = accuracy_score(y_test, y_pred)
# =============================================================================

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
acc_score = accuracy_score(y_test, y_pred)

# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'criterion': ['gini'], 
               'splitter': ['best'],
               'min_samples_split': [2],
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
