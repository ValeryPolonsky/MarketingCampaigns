# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

class ClassifierData(): 
    X_scaler: StandardScaler 
    Classifier: DecisionTreeClassifier

# Deviding dataset to training & test sets 
# =============================================================================
# dataset = pd.read_csv('data_clusters_devided.csv')
# dataset_trainig = dataset.sample(n = (int)(len(dataset.index) * 0.8))
# dataset_test = dataset[~dataset.index.isin(dataset_trainig.index)]
# 
# dataset_trainig = dataset_trainig.sample(frac=1).reset_index(drop=True)
# dataset_test = dataset_test.sample(frac=1).reset_index(drop=True)
# dataset_trainig.to_csv('data_clusters_devided_training.csv')
# dataset_test.to_csv('data_clusters_devided_test.csv')
# =============================================================================

# Importing the dataset
dataset_train = pd.read_csv('data_clusters_devided_training.csv')
dataset_test = pd.read_csv('data_clusters_devided_test.csv')
dataset_1 = dataset_train[dataset_train['deposit'] == 1]
dataset_0 = dataset_train[dataset_train['deposit'] == 0]
clusters_1 = dataset_1['cluster'].unique()
clusters_1.sort()
clusters_0 = dataset_0['cluster'].unique()
clusters_0.sort()

def CreateClassifier(X_train, y_train):    
       
    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    
    # Training the Decision Tree Classification model on the Training set
    classifier = DecisionTreeClassifier(criterion = 'entropy', 
                                        splitter='best',
                                        min_samples_split = 2,                                  
                                        random_state = 0)
    classifier.fit(X_train, y_train)
    
    classifierData = ClassifierData()
    classifierData.X_scaler = sc
    classifierData.Classifier = classifier
    
    return classifierData

# Training the Decision tree models
classifierDataList = []
for cluster_0 in clusters_0:
    for cluster_1 in clusters_1:
        dataset_clusters = dataset_train[dataset_train['cluster'].isin([cluster_0,cluster_1])]
        X = dataset_clusters.iloc[:, :-2].values
        y = dataset_clusters.iloc[:, -1].values
        classifierDataList.append(CreateClassifier(X, y))
        

def CreateNewDataSet(dataset, classifierDataList):
    new_dataset = pd.DataFrame()
    
    for row in range(0, len(dataset.index)):
        y_row_pred_list = []
        y_row_real = dataset.iloc[row][17]
        
        for classifier_data in classifierDataList:
            X_row = dataset.iloc[[row]].iloc[:,:-2].values
            X_row_scaled = classifier_data.X_scaler.transform(X_row)
            y_row_pred = classifier_data.Classifier.predict(X_row_scaled)
            y_row_pred_list.append(y_row_pred)
            
        y_row_pred_list = np.array(y_row_pred_list).reshape(1,len(y_row_pred_list))
        temp_dataset = pd.DataFrame(y_row_pred_list)
        temp_dataset[60] = y_row_real
        new_dataset = pd.concat([new_dataset, temp_dataset])
        print(f'Completed row: {row}, out of: {len(dataset.index)}')
        
    new_dataset = new_dataset.sample(frac=1).reset_index(drop=True)
    return new_dataset
 
# Creating new dataset with classifiers data   
new_dataset_train = CreateNewDataSet(dataset_train, classifierDataList)
new_dataset_train.to_csv('data_classifiers_training.csv', index = False)

new_dataset_test = CreateNewDataSet(dataset_test, classifierDataList)
new_dataset_test.to_csv('data_classifiers_test.csv', index = False)
    
    
    
    





