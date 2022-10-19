# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('data.csv')
dataset_yes = dataset[dataset["deposit"] == 'yes']
dataset_no = dataset[dataset["deposit"] == 'no']
#dataset_merged = pd.concat([dataset_yes,dataset_no])
dataset_merged = pd.concat([dataset_yes])
dataset_merged = dataset_merged.sample(frac=1).reset_index(drop=True)

X = dataset_merged.values

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
X[:,16] = le.fit_transform(X[:,16])

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

number_of_clusters = 5

# Training the K-Means model on the dataset
kmeans = KMeans(n_clusters = number_of_clusters, init = 'k-means++')
y_kmeans = kmeans.fit_predict(X)

X_means = dict()
for cluster in range (0,number_of_clusters):
    X_means[cluster] = X[y_kmeans == cluster]
    cluster_column = []
    for j in range(0,len(X_means[cluster])):
        cluster_column.append(cluster) 
    cluster_column =  np.array(cluster_column).reshape(len(cluster_column),1)
    cluster_column = cluster_column.astype(float)          
    X_means[cluster] = np.append(X_means[cluster], cluster_column, axis = 1)
          
    
# Creating new dataset with clusters
new_dataset = pd.DataFrame()
max_cluster = 0
for cluster in X_means:
    new_dataset = pd.concat([new_dataset, pd.DataFrame(X_means[cluster])])

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
                                          new_dataset.columns[16]: 'deposit',
                                          new_dataset.columns[17]: 'cluster'})
new_dataset = new_dataset.sample(frac=1).reset_index(drop=True)
for row in range (0,len(new_dataset.index)):
    new_dataset.iloc[row][16] = 1

new_dataset.to_csv('data_clusters_yes.csv')