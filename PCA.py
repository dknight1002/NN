import numpy as np
import pandas

def PCA(X , num_components):
    X_meaned = X - np.mean(X , axis = 0)
    cov_mat = np.cov(X_meaned , rowvar = False)
    eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvectors = eigen_vectors[:,sorted_index]
    eigenvector_subset = sorted_eigenvectors[:,0:num_components]
    X_reduced = np.dot(eigenvector_subset.transpose() , X_meaned.transpose() ).transpose()
    return X_reduced

dataframe = pandas.read_csv("Iris.csv", header=None)
x = dataframe.iloc[:,0:4]
target = dataframe.iloc[:,4]
mat_reduced = PCA(x , 2)
print(mat_reduced)
