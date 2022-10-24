# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

# Importing the dataset
dataset = pd.read_csv('data.csv')
dataset = dataset[dataset['deposit'] == 'no']

X = dataset.iloc[:, :-1].values

# Taking care of missing data
from sklearn.impute import SimpleImputer
numeric_columns = [0,5,9,11,12,13,14]
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, numeric_columns])
X[:, numeric_columns] = imputer.transform(X[:, numeric_columns])

from sklearn.impute import SimpleImputer
string_columns = [1,2,3,4,6,7,8,10,15]
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer.fit(X[:, string_columns])
X[:, string_columns] = imputer.transform(X[:, string_columns])

# Encoding variables
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for column in string_columns:
    X[:,column] = le.fit_transform(X[:,column])

# Training the K-Means model on the dataset
def GetClusters(X, X_means, max_cluster_size):
    kmeans = KMeans(n_clusters = 2, init = 'k-means++')
    y_kmeans = kmeans.fit_predict(X)
    
    for i in range(0,2):        
        if (len(X[y_kmeans == i]) > max_cluster_size):
            GetClusters(X[y_kmeans == i], X_means, max_cluster_size)
        else:
            cluster = 0
            while(cluster in X_means):
                cluster += 1
                
            X_means[cluster] = X[y_kmeans == i]
            
            cluster_column = []
            for j in range(0,len(X_means[cluster])):
                cluster_column.append(cluster) 
            cluster_column =  np.array(cluster_column).reshape(len(cluster_column),1)
            cluster_column = cluster_column.astype(float)
                    
            X_means[cluster] = np.append(X_means[cluster], cluster_column, axis = 1)
                      
X_means = dict()
GetClusters(X, X_means, max_cluster_size = 1000)
    
# Creating new dataset with clusters
new_dataset = pd.DataFrame()
for cluster in X_means:
    new_dataset = pd.concat([new_dataset, pd.DataFrame(X_means[cluster])])
    
for row in range (0,len(new_dataset.index)):
    new_dataset.loc[row, 16] += 6
    
deposit_value = 0
deposit_column = []
for j in range(0,len(new_dataset.index)):
    deposit_column.append(deposit_value) 
deposit_column =  np.array(deposit_column).reshape(len(deposit_column),1)
deposit_column = deposit_column.astype(int)
new_dataset[17] = deposit_column
    
new_dataset = new_dataset.rename(columns={new_dataset.columns[0]: 'age',
                                          new_dataset.columns[1]: 'job',
                                          new_dataset.columns[2]: 'marital',
                                          new_dataset.columns[3]: 'education',
                                          new_dataset.columns[4]: 'default',
                                          new_dataset.columns[5]: 'balance',
                                          new_dataset.columns[6]: 'housing',
                                          new_dataset.columns[7]: 'loan',
                                          new_dataset.columns[8]: 'contact',
                                          new_dataset.columns[9]: 'day',
                                          new_dataset.columns[10]: 'month',
                                          new_dataset.columns[11]: 'duration',
                                          new_dataset.columns[12]: 'campaign',
                                          new_dataset.columns[13]: 'pdays',
                                          new_dataset.columns[14]: 'previous',
                                          new_dataset.columns[15]: 'poutcome',
                                          new_dataset.columns[16]: 'cluster',
                                          new_dataset.columns[17]: 'deposit'})
new_dataset = new_dataset.sample(frac=1).reset_index(drop=True)
new_dataset.to_csv('data_clusters_no.csv')


dataset = pd.concat([pd.read_csv('data_clusters_yes.csv'), pd.read_csv('data_clusters_no.csv')])
dataset = dataset.sample(frac=1).reset_index(drop=True)
dataset.to_csv('data_clusters_devided.csv')