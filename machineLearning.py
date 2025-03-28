import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


''' PRE PROCESSING '''
def splitData(df, label_list, randomness, split_value):

  train_data, test_data, train_labels, test_labels = train_test_split(
    df, label_list, test_size=split_value, random_state=randomness, stratify=label_list
  )

  return train_data, test_data, train_labels, test_labels

def scaleFeatures(df):
  scaler = StandardScaler()
  scaler.set_output(transform="pandas")

  scaled_features = scaler.fit_transform(df)

  return scaled_features

''' PCA '''
def setHyperparams(kfold_train_data_scaled, variance_explained):
    
    C = np.cov(kfold_train_data_scaled, rowvar=False) # 140x140 Co-variance matrix
    eigenvalues, eigenvectors = np.linalg.eig(C)

    eig_sum = 0
    for i in range(len(eigenvalues)):
        
        eig_sum += eigenvalues[i]
        total_variance = eig_sum / eigenvalues.sum()

        if total_variance >= variance_explained:
            n_components = i + 1
            print(f"Variance explained by {i + 1} PCA components: {eig_sum / eigenvalues.sum()}")
            break

    n_components = 5

    return n_components