import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV, HalvingGridSearchCV, RandomizedSearchCV
from sklearn.decomposition import PCA
from sklearn import svm, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns

from plotting import biplot


''' PRE PROCESSING '''

def trainScaler(df):
  scaler = StandardScaler()
  scaler.set_output(transform="pandas")

  scaler.fit(df)

  return scaler

''' PCA '''
def setNComponents(kfold_train_data_scaled, variance_explained):
    if variance_explained < 1.0:
      C = np.cov(kfold_train_data_scaled, rowvar=False) # 140x140 Co-variance matrix
      eigenvalues, eigenvectors = np.linalg.eig(C)

      eig_sum = 0
      for i in range(len(eigenvalues)):
          
          eig_sum += eigenvalues[i]
          total_variance = eig_sum / eigenvalues.sum()

          if total_variance >= variance_explained:
              n_components = i + 1
              print(f"Variance explained by {n_components} PCA components: {eig_sum / eigenvalues.sum()}")
              # print(eigenvalues[0] / eigenvalues.sum())
              # print((eigenvalues[0] + eigenvalues[1]) / eigenvalues.sum())
              # print((eigenvalues[0] + eigenvalues[1] + eigenvalues[2]) / eigenvalues.sum())
              # print((eigenvalues[0] + eigenvalues[1] + eigenvalues[2] + eigenvalues[3]) / eigenvalues.sum())
              # print((eigenvalues[0] + eigenvalues[1] + eigenvalues[2] + eigenvalues[3] + eigenvalues[4]) / eigenvalues.sum())
              break
          
    else:
       n_components = variance_explained
    
    return n_components

def makeSmoothParamGrid(param_grid):
   
  smooth_param_grid = {}

  for param, value in param_grid.items():

    if value and all(v is None for v in value):
       smooth_param_grid[param] = None
       continue
    
    if all(isinstance(v, (str, bool)) for v in value):
      # print(f"  - Converted '{param}' to Categorical({value})")
      smooth_param_grid[param] = Categorical(value)
      continue
    
    # Filter out None
    if any(isinstance(v, float) for v in value):
      float_values = [float(v) for v in value if v != None]
      min_value, max_value = min(float_values), max(float_values)

      if abs(max_value - min_value) > 1e-12:
        # print(f"  - Converted '{param}' to Real({value})")
        smooth_param_grid[param] = Real(min_value, max_value, name=param)

    if any(isinstance(v, int) for v in value):
        int_values = [float(v) for v in value if v != None]
        min_value, max_value = min(int_values), max(int_values)

        if abs(max_value - min_value) > 1e-12:
          # print(f"  - Converted '{param}' to Integer({value})")
          smooth_param_grid[param] = Integer(int(min_value), int(max_value), name=param)

  return smooth_param_grid

def makeSVMClassifier(method, base_estimator, num_folds, param_grid, df, labels, train_data, variance_explained):
    
    print()
    print(f"Classifier: \t {base_estimator}")
    print(f"Optimalizer: \t {method}")
    print("-" * 40)

    start_time = time.time()

    if method.lower() == 'manualgridsearchcv':
      
      # Unpack dictionary into lists
      C_list, kernel_types, gamma_list, coef0_list, deg_list = [list(values) for values in param_grid.values()]

      hyperparams_list = []

      # Initialize arrays for evaluating score = (mean - std)
      metrics_matrix = np.zeros( (num_folds, len(C_list), len(kernel_types), len(gamma_list), len(coef0_list), len(deg_list)) )
      metrics_matrix_mean = np.zeros( (len(C_list), len(kernel_types), len(gamma_list), len(coef0_list), len(deg_list)) )
      metrics_matrix_std = np.zeros( (len(C_list), len(kernel_types), len(gamma_list), len(coef0_list), len(deg_list)) )

      ''' K-FOLD SPLIT '''

      skf = StratifiedKFold(n_splits = num_folds)

      ''' HYPERPARAMETER OPTIMIZATION '''
      for i, (train_index, test_index) in enumerate(skf.split(train_data, labels)):

        print(f"PCA fitting on fold {i}")
          
        # Debug prints
        # print(f"  Train: index={train_index}")
        # print(f"  Test:  index={test_index}")
        # print(f"Train labels: {train_labels}")

        kfold_train_labels = [labels[j] for j in train_index]
        kfold_test_labels = [labels[j] for j in test_index]

        # unique, counts = np.unique(kfold_train_labels, return_counts=True)
        # print(dict(zip(unique, counts)))

        # print(kfold_testLabels)
        
        kfold_train_data = train_data.iloc[train_index]
        kfold_validation_data = train_data.iloc[test_index]

        # Scale training and validation separately
        scaler = StandardScaler()
        scaler.set_output(transform="pandas")

        kfold_train_data_scaled = scaler.fit_transform(kfold_train_data)
        kfold_validation_data_scaled = scaler.transform(kfold_validation_data)

        PCA_components = setNComponents(kfold_train_data_scaled, variance_explained=variance_explained)
        
        PCA_fold = PCA(n_components = PCA_components)
        
        kfold_PCA_train_df = pd.DataFrame(PCA_fold.fit_transform(kfold_train_data_scaled))
        kfold_PCA_validation_df = pd.DataFrame(PCA_fold.transform(kfold_validation_data_scaled))

        # if (want_plots):
        #   print(f"Plotting PCA plots for fold {i}")
        #   biplot(kfold_PCA_train_df, kfold_train_labels, PCA_fold, PCA_components, separate_types)


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

      print(f"Created {len(hyperparams_list) * num_folds} SVM classifiers.")
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


      clf_best_params = {
          "C": C_list[multi_dim_index[0]],
          "kernel": kernel_types[multi_dim_index[1]],
          "gamma": gamma_list[multi_dim_index[2]],
          "coef0": coef0_list[multi_dim_index[3]],
          "degree": deg_list[multi_dim_index[4]]
      }
        
      print(f"All combinations of hyper params: {len(hyperparams_list)}")

      clf = svm.SVC(**clf_best_params)
      clf.fit(df, labels)

      end_time = time.time()  # End timer
      elapsed_time = end_time - start_time
      
      print(f"Best model found and fitted in {elapsed_time:.4f} seconds")
      print(f"{clf_best_params} gives the parameter setting with the highest (mean-std) score: {max_value}")
      print(f"\n")

    elif method.lower() == 'gridsearchcv':

      clf = GridSearchCV(
            estimator = base_estimator,
            param_grid = param_grid,
            scoring = 'accuracy',
            cv = num_folds, 
            verbose = 0,
            n_jobs = -1
            )
      
      clf.fit(df, labels)
      clf_best_params = clf.best_params_

      end_time = time.time()  # End timer
      elapsed_time = end_time - start_time

      best_score = max( clf.cv_results_['mean_test_score'] - clf.cv_results_['std_test_score'] )

      print(f"Best model found and fitted in {elapsed_time:.4f} seconds")
      print(f"{clf.cv_results_['params'][clf.best_index_]} gives the parameter setting with the highest (mean - std): {best_score}")
      print(f"\n")

    elif method.lower() == 'halvinggridsearchcv':
     
      clf = HalvingGridSearchCV(
            estimator = base_estimator,
            param_grid = param_grid,
            factor = 2,
            scoring = 'accuracy',
            cv = num_folds,
            verbose = 0,
            n_jobs = -1
            )
      
      clf.fit(df, labels)
      clf_best_params = clf.best_params_

      end_time = time.time()  # End timer
      elapsed_time = end_time - start_time

      best_score = max( clf.cv_results_['mean_test_score'] - clf.cv_results_['std_test_score'] )

      print(f"Best model found and fitted in {elapsed_time:.4f} seconds")
      print(f"{clf.cv_results_['params'][clf.best_index_]} gives the parameter setting with the highest (mean - std): {best_score}")
      print(f"\n")

    elif method.lower() == 'bayessearchcv':
      
      hyperparams_space = {
        "C": Real(param_grid['C'][0], param_grid['C'][-1], prior="log-uniform"),  # Continuous log-scale for C
        "kernel": Categorical(["linear", "poly", "rbf", "sigmoid"]),  # Discrete choices
        "gamma": Real(param_grid['gamma'][0], param_grid['gamma'][-1], prior="log-uniform"),  # Log-uniform scale for gamma
        "coef0": Real(param_grid['coef0'][0], param_grid['coef0'][-1]),
        "degree": Integer(param_grid['degree'][0], param_grid['degree'][-1])
      }

      clf = BayesSearchCV(
            estimator = base_estimator,
            search_spaces = hyperparams_space,
            n_iter = 30,
            scoring = 'accuracy',
            cv = num_folds,
            verbose = 0,
            n_jobs = 5
            )
      
      clf.fit(df, labels)
      clf_best_params = clf.best_params_

      end_time = time.time()  # End timer
      elapsed_time = end_time - start_time

      best_score = max( clf.cv_results_['mean_test_score'] - clf.cv_results_['std_test_score'] )

      print(f"Best model found and fitted in {elapsed_time:.4f} seconds")
      print(f"{clf.cv_results_['params'][clf.best_index_]} gives the parameter setting with the highest (mean - std): {best_score}")
      print(f"\n")

    elif method.lower() == 'randomizedsearchcv':
      
      clf = RandomizedSearchCV(
            estimator = base_estimator,
            param_distributions = param_grid,
            n_iter = 30,
            scoring = 'accuracy',
            cv = num_folds,
            verbose = 0,
            n_jobs = 5
            )
      
      clf.fit(df, labels)
      clf_best_params = clf.best_params_

      end_time = time.time()  # End timer
      elapsed_time = end_time - start_time

      best_score = max( clf.cv_results_['mean_test_score'] - clf.cv_results_['std_test_score'] )

      print(f"Best model found and fitted in {elapsed_time:.4f} seconds")
      print(f"{clf.cv_results_['params'][clf.best_index_]} gives the parameter setting with the highest (mean - std): {best_score}")
      print(f"\n")
      
    else: 
      
      clf = base_estimator
      clf.fit(df, labels)
      clf_best_params = {'C':1, 'kernel': 'rbf'}

      end_time = time.time()  # End timer
      elapsed_time = end_time - start_time

      print(f"Base model fitted in {elapsed_time:.4f} seconds")
      print(f"Optimizer {method} not recognized, using default Support Vector Classifier.")
      print(f"\n")

    return clf, clf_best_params

def makeRFClassifier(method, base_estimator, num_folds, param_grid, df, labels):
  
  print()
  print(f"Classifier: \t {base_estimator}")
  print(f"Optimalizer: \t {method}")
  print("-" * 40)
  
  start_time = time.time()
  
  if method == 'GridSearchCV':
    
    clf = GridSearchCV(estimator=base_estimator,
                       param_grid=param_grid,
                       cv=num_folds,
                       n_jobs=-1
                       ) 
    
    clf.fit(df, labels)
    clf_best_params = clf.best_params_

    end_time = time.time()  # End timer
    elapsed_time = end_time - start_time

    best_score = max( clf.cv_results_['mean_test_score'] - clf.cv_results_['std_test_score'] )

    print(f"Best model found and fitted in {elapsed_time:.4f} seconds")
    print(f"{clf.cv_results_['params'][clf.best_index_]} gives the parameter setting with the highest (mean - std): {best_score}")
    print(f"\n")

  elif method == 'HalvingGridSearchCV':

    clf = HalvingGridSearchCV(
          estimator=base_estimator,
          param_grid = param_grid,
          cv=num_folds,
          n_jobs=-1
          )
    
    clf.fit(df, labels)
    clf_best_params = clf.best_params_
    
    end_time = time.time()  # End timer
    elapsed_time = end_time - start_time

    best_score = max( clf.cv_results_['mean_test_score'] - clf.cv_results_['std_test_score'] )

    print(f"Best model found and fitted in {elapsed_time:.4f} seconds")
    print(f"{clf.cv_results_['params'][clf.best_index_]} gives the parameter setting with the highest (mean - std): {best_score}")
    print(f"\n")

  else:
    
    clf = base_estimator

    clf.fit(df, labels)
    clf_best_params = {'n_estimators': 100,
                        'criterion': 'gini',
                        'max_depth': None,
                        'min_samples_split': 2,
                        'min_samples_leaf': 1,
                        'max_features': 'sqrt'}
  
    end_time = time.time()  # End timer
    elapsed_time = end_time - start_time

    print(f"Base model fitted in {elapsed_time:.4f} seconds")
    print(f"Method {method} not recognized, fitting default Random Forest Classifier")
    print(f"\n")

  return clf, clf_best_params

def makeKNNClassifier(method, base_estimator, num_folds, param_grid, df, labels):
    
    print()
    print(f"Classifier: \t {base_estimator}")
    print(f"Optimalizer: \t {method}")
    print("-" * 40)

    start_time = time.time()

    if method.lower() == 'gridsearchcv':
        
        clf = GridSearchCV(
            estimator=base_estimator,
            param_grid=param_grid,
            cv=num_folds,
            n_jobs=-1
            ) 
        
        clf.fit(df, labels)
        clf_best_params = clf.best_params_

        end_time = time.time()  # End timer
        elapsed_time = end_time - start_time

        best_score = max( clf.cv_results_['mean_test_score'] - clf.cv_results_['std_test_score'] )

        print(f"Best model found and fitted in {elapsed_time:.4f} seconds")
        print(f"{clf.cv_results_['params'][clf.best_index_]} gives the parameter setting with the highest (mean - std): {best_score}")
        print(f"\n")

    elif method.lower() == 'halvinggridsearchcv':
        
        clf = HalvingGridSearchCV(
            estimator=base_estimator,
            param_grid=param_grid,
            cv=num_folds,
            n_jobs=-1
            ) 
        
        clf.fit(df, labels)
        clf_best_params = clf.best_params_

        end_time = time.time()  # End timer
        elapsed_time = end_time - start_time

        best_score = max( clf.cv_results_['mean_test_score'] - clf.cv_results_['std_test_score'] )

        print(f"Best model found and fitted in {elapsed_time:.4f} seconds")
        print(f"{clf.cv_results_['params'][clf.best_index_]} gives the parameter setting with the highest (mean - std): {best_score}")
        print(f"\n")

    elif method.lower() == 'randomizedsearchcv':
        
        clf = RandomizedSearchCV(
            estimator = base_estimator,
            param_distributions = param_grid,
            n_iter = 30,
            scoring = 'accuracy',
            cv = num_folds,
            verbose = 0,
            n_jobs = -1
            )
        
        clf.fit(df, labels)
        clf_best_params = clf.best_params_

        end_time = time.time()  # End timer
        elapsed_time = end_time - start_time

        best_score = max( clf.cv_results_['mean_test_score'] - clf.cv_results_['std_test_score'] )

        print(f"Best model found and fitted in {elapsed_time:.4f} seconds")
        print(f"{clf.cv_results_['params'][clf.best_index_]} gives the parameter setting with the highest (mean - std): {best_score}")
        print(f"\n")

    else:
        
        clf = KNeighborsClassifier(n_neighbors=3)

        clf.fit(df, labels)
        clf_best_params = {"n_neighbors": 3}

        end_time = time.time()  # End timer
        elapsed_time = end_time - start_time

        print(f"Base model fitted in {elapsed_time:.4f} seconds")
        print(f"Optimizer {method} not recognized, fitting default KNN model with 3 neighbors")
        print(f"\n")
    
    return clf, clf_best_params

def makeGNBClassifier(method, base_estimator, num_folds, param_grid, df, labels):
    
    print()
    print(f"Classifier: \t {base_estimator}")
    print(f"Optimalizer: \t {method}")
    print("-" * 40)
    
    if (method == "ahadhaidiahodihaj"):
        print("HOW?!")
    else:
        clf = base_estimator
        clf.fit(df, labels)

        print(clf.get_params())
    return clf

def evaluateCLF(name, clf, test_df, test_labels, want_plots, activity_name, clf_name):
    
    print(f"{name} scores")

    test_predict = clf.predict(test_df)

    accuracy_score = metrics.accuracy_score(test_labels, test_predict)
    precision_score = metrics.precision_score(test_labels, test_predict, average="weighted")
    recall_score = metrics.recall_score(test_labels, test_predict, average="weighted")
    f1_score = metrics.f1_score(test_labels, test_predict, average="weighted")

    print(f"Accuracy: \t {accuracy_score:.4f}")
    print(f"Precision: \t {precision_score:.4f}")
    print(f"Recall: \t {recall_score:.4f}")
    print(f"f1: \t\t {f1_score:.4f}")
    print("-" * 23)

    if(want_plots):
        ''' CONFUSION MATRIX '''
        conf_matrix = metrics.confusion_matrix(test_labels, test_predict, labels=activity_name)
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, cmap='coolwarm', xticklabels=activity_name, yticklabels=activity_name)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f'Confusion matrix, {clf_name}, {name}')
    
    return accuracy_score

def makeClassifier(base_estimator, param_grid, method, X, y, search_kwargs,  n_iter=30):
    
    print()
    print(f"Classifier: \t {base_estimator}")
    print(f"Optimalizer: \t {method}")
    print("-" * 40)

    start_time = time.time()

    if method.lower() == 'gridsearchcv':
        
        clf = GridSearchCV(
           
            estimator=base_estimator,
            param_grid=param_grid,
            
            **search_kwargs

            ) 

    elif method.lower() == 'halvinggridsearchcv':
        
        clf = HalvingGridSearchCV(
           
            estimator=base_estimator,
            param_grid=param_grid,
           
            **search_kwargs

            ) 
         
    elif method.lower() == 'randomizedsearchcv':
        
        clf = RandomizedSearchCV(
           
            estimator=base_estimator,
            param_distributions=param_grid,
            
            n_iter=n_iter,
            **search_kwargs

            ) 
        
    elif method.lower() == 'bayessearchcv':
        
        smooth_param_grid = makeSmoothParamGrid(param_grid)

        clf = BayesSearchCV(

            estimator=base_estimator,
            search_spaces=smooth_param_grid,
          
            n_iter=n_iter,
            **search_kwargs

            )

    else:
       clf = base_estimator
       best_params = None
       print(f"{method} not recognized, fitting default {base_estimator}")

    clf.fit(X, y)

    end_time = time.time()  # End timer
    elapsed_time = end_time - start_time

    if clf != base_estimator: 
      best_params = clf.best_params_

      best_score = ( clf.cv_results_['mean_test_score'][clf.best_index_] - clf.cv_results_['std_test_score'][clf.best_index_] )
      print(clf.best_score_)
      print(f"{clf.cv_results_['params'][clf.best_index_]} gives the parameter setting with the highest (mean - std): {best_score}")
      print(f"Best model found and fitted in {elapsed_time:.4f} seconds")
      print(f"\n")  

    return clf, best_params