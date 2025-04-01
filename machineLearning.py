import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from skopt import BayesSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV, HalvingGridSearchCV
from sklearn.decomposition import PCA
from sklearn import svm, metrics

from plotting import biplot


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
def setNComponents(kfold_train_data_scaled, variance_explained):
    
    C = np.cov(kfold_train_data_scaled, rowvar=False) # 140x140 Co-variance matrix
    eigenvalues, eigenvectors = np.linalg.eig(C)

    eig_sum = 0
    for i in range(len(eigenvalues)):
        
        eig_sum += eigenvalues[i]
        total_variance = eig_sum / eigenvalues.sum()

        if total_variance >= variance_explained:
            n_components = i + 1
            print(f"Variance explained by {n_components} PCA components: {eig_sum / eigenvalues.sum()}")
            # print(f"Varaince explained by {eigenvalues[0]/eigenvalues.sum()}")
            # print(f"Varaince explained by {eigenvalues[1]/eigenvalues.sum()}")
            # print(f"Varaince explained by {eigenvalues[2]/eigenvalues.sum()}")
            # print(f"Varaince explained by {eigenvalues[3]/eigenvalues.sum()}")
            # print(f"Varaince explained by {eigenvalues[4]/eigenvalues.sum()}")
            break
    
    return n_components

def makeSVMClassifier(method, num_folds, hyperparams_space, hyperparams_dict, want_plots, PCA_train_df, train_data, train_labels, variance_explained, seperate_types):
    
    # Unpack dictionary into lists
    C_list, kernel_types, gamma_list, coef0_list, deg_list = [list(values) for values in hyperparams_dict.values()]
    hyperparams_list = []

    start_time = time.time()

    if method == 'ManualGridSearch':

      # Initialize arrays for evaluating score = (mean - std)
      metrics_matrix = np.zeros( (num_folds, len(C_list), len(kernel_types), len(gamma_list), len(coef0_list), len(deg_list)) )
      metrics_matrix_mean = np.zeros( (len(C_list), len(kernel_types), len(gamma_list), len(coef0_list), len(deg_list)) )
      metrics_matrix_std = np.zeros( (len(C_list), len(kernel_types), len(gamma_list), len(coef0_list), len(deg_list)) )

      ''' K-FOLD SPLIT '''

      skf = StratifiedKFold(n_splits = num_folds)

      ''' HYPERPARAMETER OPTIMIZATION '''
      for i, (train_index, test_index) in enumerate(skf.split(train_data, train_labels)):

        print(f"PCA fitting on fold {i}")
          
        # Debug prints
        # print(f"  Train: index={train_index}")
        # print(f"  Test:  index={test_index}")
        # print(f"Train labels: {train_labels}")

        kfold_train_labels = [train_labels[j] for j in train_index]
        kfold_test_labels = [train_labels[j] for j in test_index]

        # unique, counts = np.unique(kfold_train_labels, return_counts=True)
        # print(dict(zip(unique, counts)))

        # print(kfold_testLabels)
        
        kfold_train_data = train_data.iloc[train_index]
        kfold_validation_data = train_data.iloc[test_index]

        # Scale training and validation separately
        kfold_train_data_scaled = scaleFeatures(kfold_train_data)
        kfold_validation_data_scaled = scaleFeatures(kfold_validation_data)
        
        PCA_components = setNComponents(kfold_train_data_scaled, variance_explained=variance_explained)
        
        PCA_fold = PCA(n_components = PCA_components)
        
        kfold_PCA_train_df = pd.DataFrame(PCA_fold.fit_transform(kfold_train_data_scaled))
        kfold_PCA_validation_df = pd.DataFrame(PCA_fold.transform(kfold_validation_data_scaled))

        if (want_plots):
          print(f"Plotting PCA plots for fold {i}")
          biplot(kfold_PCA_train_df, kfold_train_labels, PCA_fold, PCA_components, seperate_types)


        for j, C_value in enumerate(C_list):

            for k, kernel in enumerate(kernel_types):
                    
                # print("Work in progress")

              if kernel == 'linear':
                l, m, n = 0, 0, 0
                      
                clf = svm.SVC(C=C_value, kernel=kernel)
                clf.fit(kfold_PCA_train_df, kfold_train_labels)
                test_predict = clf.predict(kfold_PCA_validation_df)
                # accuracy_array[i, j, k, :, :, :] = 0
                metrics_matrix[i, j, k, l, m, n] = metrics.f1_score(kfold_test_labels, test_predict, average="micro")
                
                # Only append for fold 0
                if i == 0:
                    hyperparams_list.append((C_value, kernel))

              elif kernel == 'poly':
                      
                for l, gamma_value in enumerate(gamma_list):
                    for m, coef0_value in enumerate(coef0_list):
                        for n, deg_value in enumerate(deg_list):

                          # print(f"Working on {j} {k} {l} {m} {n}")
                          clf = svm.SVC(C=C_value, kernel=kernel, gamma=gamma_value, coef0=coef0_value, degree=deg_value)
                          clf.fit(kfold_PCA_train_df, kfold_train_labels)
                          test_predict = clf.predict(kfold_PCA_validation_df)
                          metrics_matrix[i, j, k, l, m, n] = metrics.f1_score(kfold_test_labels, test_predict, average="micro")
                          
                          if i == 0:
                              hyperparams_list.append((C_value, kernel, gamma_value, coef0_value, deg_value))

              elif kernel == 'sigmoid': 
                  
                for l, gamma_value in enumerate(gamma_list):
                    for m, coef0_value in enumerate(coef0_list):    
                        n = 0
                        clf = svm.SVC(C=C_value, kernel=kernel, gamma=gamma_value, coef0=coef0_value)
                        clf.fit(kfold_PCA_train_df, kfold_train_labels)
                        test_predict = clf.predict(kfold_PCA_validation_df)
                        metrics_matrix[i, j, k, l, m, n] = metrics.f1_score(kfold_test_labels, test_predict, average="micro")   

                        if i == 0:
                            hyperparams_list.append((C_value, kernel, gamma_value, coef0_value))

              elif kernel == 'rbf':

                for l, gamma_value in enumerate(gamma_list):
                    m, n = 0, 0
                    clf = svm.SVC(C=C_value, kernel=kernel, gamma=gamma_value)
                    clf.fit(kfold_PCA_train_df, kfold_train_labels)
                    test_predict = clf.predict(kfold_PCA_validation_df)
                    metrics_matrix[i, j, k, l, m, n] = metrics.f1_score(kfold_test_labels, test_predict, average="micro")  

                    if i == 0:
                        hyperparams_list.append((C_value, kernel, gamma_value))

      print("\n")

      # Exhaustive grid search: calculate which hyperparams gives highest score = max|mean - std|
      for j in range(len(C_list)):
          for k in range(len(kernel_types)):
              for l in range(len(gamma_list)):
                  for m in range(len(coef0_list)):
                      for n in range(len(deg_list)):
                          metrics_matrix_mean[j, k, l, m, n] = metrics_matrix[:, j, k, l, m, n].mean()
                          metrics_matrix_std[j, k, l, m, n] = metrics_matrix[:, j, k, l, m, n].std()

      score_matrix = metrics_matrix_mean - metrics_matrix_std

      # Find location and value of highest score
      max_value_index = np.argmax(score_matrix)
      max_value = np.max(score_matrix)
      multi_dim_index = np.unravel_index(max_value_index, score_matrix.shape)
      # print(score_array.shape)
      # print(best_param)
      # print(len(multi_dim_index))

      # Unknown error !!!!!!!!!!!!!!! ^ must be investigated

      '''
      Tror problemet ligger i at score_array er en cube med mange 0 verdier
      0 for alle verdier som ikke settes i loopen (degree 2,3,4 for 'linear' fr.eks)
      Potensielt bevis: score_array uten alle 0 verdier er like lang som hyper_param_list
      Videre gir multi_dim_index og hyper_param_list(best_param_test) like parametre
      '''
      # score_array_test = score_matrix.flatten()
      # score_array_test = [i for i in score_array_test if i != 0]
      # # print(score_array_1D)
      # print(f"Size of score_array_test: {len(score_array_test)}")
      # # Find location and value of highest score
      # best_param = np.argmax(score_matrix)
      # print(f"Index of best parameter, converted to 2D array (cube): {best_param}")
      # best_param_test = np.argmax(score_array_test) 
      # print(f"Index of best parameter, converted to 2D array (not cube): {best_param_test}")
      # print(f"Best combination of hyperparameters (C, kernel, gamma, coef0, degree): {hyperparams_list[best_param_test]}")
      # ''' Ser en del endringer ble gjort, legger denne her for nå, kan slettes '''

      print(f"\n")
      print(f"All combinations of hyper params: {len(hyperparams_list)}")
      print(f"Highest score found (mean - std): {max_value}")
      

      best_hyperparams = {
          "C": C_list[multi_dim_index[0]],
          "kernel": kernel_types[multi_dim_index[1]],
          "gamma": gamma_list[multi_dim_index[2]],
          "coef0": coef0_list[multi_dim_index[3]],
          "degree": deg_list[multi_dim_index[4]]
      }

      print(f"Using ManualGridSearch to find best hyperparams: {best_hyperparams}")  

      if(want_plots):
        plt.show()

      clf = svm.SVC(**best_hyperparams)
      clf.fit(PCA_train_df, train_labels)


    elif method == 'GridSearchCV':

      print(f"Using GridSearchCV from sklearn to find best hyperparams")

      clf = GridSearchCV(
            estimator = svm.SVC(),
            param_grid = hyperparams_dict,
            scoring = 'accuracy',
            cv = num_folds, 
            verbose = 0,
            n_jobs = -1
            )
      
      clf.fit(PCA_train_df, train_labels)

      for param, value in clf.best_params_.items():
        print(f"{param}: {value}")
      
    elif method == 'HalvingGridSearchCV':
      
      print(f"Using HalvingGridSearchCV from sklearn to find best hyperparams")

      clf = HalvingGridSearchCV(
            estimator = svm.SVC(),
            param_grid = hyperparams_dict,
            factor = 2,
            scoring = 'accuracy',
            cv = num_folds,
            verbose = 0,
            n_jobs = -1
            )
      
      clf.fit(PCA_train_df, train_labels)

      for param, value in clf.best_params_.items():
        print(f"{param}: {value}")

      
    elif method == 'BayesSearchCV':
      
      print(f"Using BayesSearchCV from scikit optimize to find best hyperparams")

      clf = BayesSearchCV(
            estimator = svm.SVC(),
            search_spaces = hyperparams_space,
            n_iter = 30,
            scoring = 'accuracy',
            cv = num_folds,
            verbose = 0,
            n_jobs = -1
            )
      
      clf.fit(PCA_train_df, train_labels)
    
      for param, value in clf.best_params_.items():
        print(f"{param}: {value}")

    else: 
      print(f"Optimizer {method} not recognized, choosing default Support Vector Classifier.")
      clf = svm.SVC()
      clf.fit(PCA_train_df, train_labels)
      return clf


    end_time = time.time()  # End timer
    elapsed_time = end_time - start_time

    print(f"Created and evaluated {len(hyperparams_list) * num_folds} instances of SVM classifiers in {elapsed_time} seconds")
    print(f"\n")

    return clf


def makeRFClassifier(method, num_folds, hyperparams_space, hyperparams_dict, want_plots, PCA_train_df, train_data, train_labels, variance_explained, seperate_types):
  from sklearn.ensemble import RandomForestClassifier
  
  
  if method == 'GridSearchCV':

    print(f"Using GridSearchCV from sklearn to find best hyperparams")

    param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
       }
    

   
    clf = GridSearchCV(estimator=RandomForestClassifier(),
                       param_grid=param_grid,
                       cv=5) 
    
    
    clf.fit(PCA_train_df, train_labels)

    for param, value in clf.best_params_.items():
      print(f"{param}: {value}")
    
  elif method == 'HalvingGridSearchCV':
    print(f"Using HalvingGridSearchCV from sklearn to find best hyperparams")

    param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
       }

    clf = HalvingGridSearchCV(
          estimator=RandomForestClassifier(),
          param_grid = param_grid,
          cv=5
          )
    
    clf.fit(PCA_train_df, train_labels)

    for param, value in clf.best_params_.items():
      print(f"{param}: {value}")

def makeNai