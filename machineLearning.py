import numpy as np
import pandas as pd
import time
import joblib

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn import metrics, dummy
from typing import List, Dict, Any, Tuple, Sequence

def trainScaler(df):
  ''' PRE PROCESSING '''
  scaler = StandardScaler()
  scaler.set_output(transform="pandas")

  scaler.fit(df)

  return scaler

def setNComponents(X_train:             pd.DataFrame, 
                   variance_explained:  Any
                   ) -> int:
    
    '''
    Determines the number of PCA components needed to explain a target variance.
    '''

    if variance_explained < 1.0:
      C = np.cov(X_train, rowvar=False) # 140x140 Co-variance matrix
      eigenvalues, eigenvectors = np.linalg.eig(C)

      eig_sum = 0
      for i in range(len(eigenvalues)):
          
        eig_sum += eigenvalues[i]
        total_variance = eig_sum / eigenvalues.sum()

        if total_variance >= variance_explained:
            n_components = i + 1
            print(f"Variance explained by {n_components} PCA components: {eig_sum / eigenvalues.sum()}")
            break
          
    else:
      n_components = variance_explained
    
    return n_components

def makeSmoothParamGrid(param_grid: Dict[str, Any]
                        ) -> Dict[str, Any]:
  
  '''
  Converts a scikit-learn style parameter grid into a scikit-optimize search space.

  Iterates through a dictionary where keys are hyperparameter names and values
  are lists of potential discrete values. It converts these lists into
  appropriate scikit-optimize `Dimension` objects (`Categorical`, `Real`,
  `Integer`) based on the data types present in the list.

  This is typically used to prepare search spaces for Bayesian optimization
  methods like `skopt.BayesSearchCV`.
  '''

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

def makeNClassifiers(models:                Dict[str, Tuple[Any, Dict]],
                     model_names:           Dict[str, str],
                     optimization_methods:  List[str],
                     model_selection:       List[str],
                     method_selection:      List[str],    
                     X:                     pd.DataFrame,
                     y:                     List[str],
                     search_kwargs:         Dict[str, Any],
                     n_iter:                int,
                     BASE_MODEL_KEY:        str = 'SVM',
                     BASE_METHOD:           str = 'Base model',
                     num_of_STD:            int = 1
                     ) -> Dict[str, Any]:
  
  '''
  Generates and fits multiple scikit-learn classifiers using specified methods.

  Iterates through selected base models and hyperparameter optimization
  methods. For each combination, it instantiates and fits the appropriate
  scikit-learn hyperparameter search object (e.g., GridSearchCV,
  RandomizedSearchCV) or fits the base estimator directly if the
  optimization method is not recognized (uses the last method in
  optimization_methods as a fallback label in this case, but fits base).

  It extracts performance metrics and best parameters from the fitted
  search objects and collects them into a results list. Progress and
  key results are printed to the console during execution.
  '''

  selected_model_data   = []
  selected_model_names  = []
  results               = [] 
  method_selection_list = []

  print()


  # --- 1. Compare selected models and methods to full list ---
  # Check selected models
  if not model_selection:
    model_selection.append(BASE_MODEL_KEY)
    print(f"Warning: No model selected, defaulting to {BASE_MODEL_KEY}")

  for key in model_selection:

    key_upper = key.upper()

    estimator, grid = models.get(key_upper, models[BASE_MODEL_KEY])
    model_name_full = model_names.get(key_upper, model_names[BASE_MODEL_KEY])

    if key_upper not in models:
      print(f"Warning: Classifier '{key}' not recognized, defaulting to {model_name_full}.")

    # Create a list of (base model, param grid) pairs based on input models, default to SVM base and SVM param grid
    selected_model_names.append(model_name_full)
    selected_model_data.append((estimator, grid))

  # Check selected methods
  if not method_selection:
    method_selection.append(BASE_METHOD)

  for key in method_selection:
    
    key_upper = key.upper()

    method = optimization_methods.get(key_upper, BASE_METHOD)

    if key_upper not in optimization_methods:
      print(f"Warning: Method '{key}' not recognized, defaulting to {method}.")
    
    method_selection_list.append(method)


  # --- 2. Make n classifiers = len(model_selection) * len(method_selection) ---
  for (base_estimator, param_grid), model_name_str in zip(selected_model_data, selected_model_names):
    for method in method_selection_list:

      print()
      print(f"Classifier: \t {model_name_str}")
      print(f"Optimalizer: \t {method}")
      print("-" * 40)

      start_time = time.time()

      if method.lower() == 'gridsearchcv':

        try:
            clf = GridSearchCV(
                
                estimator=base_estimator,
                param_grid=param_grid,

                **search_kwargs,

                )
            
        except Exception as clf_error:
          print(f"Error in makeNClassifiers: {clf_error}")
          continue

      elif method.lower() == 'halvinggridsearchcv':
        
        try:
          clf = HalvingGridSearchCV(
              
              estimator=base_estimator,
              param_grid=param_grid,

              **search_kwargs

              ) 
          
        except Exception as clf_error:
          print(f"Error in makeNClassifiers: {clf_error}")
          continue
              
      elif method.lower() == 'randomizedsearchcv':
          
        try:
          clf = RandomizedSearchCV(
              
              estimator=base_estimator,
              param_distributions=param_grid,

              n_iter=n_iter,
              **search_kwargs

              ) 
          
        except Exception as clf_error:
          print(f"Error in makeNClassifiers: {clf_error}")
          continue
          
      elif method.lower() == 'bayessearchcv':
          
        try:
          smooth_param_grid = makeSmoothParamGrid(param_grid)

          clf = BayesSearchCV(

              estimator=base_estimator,
              search_spaces=smooth_param_grid,

              n_iter=n_iter,
              **search_kwargs

              )
          
        except Exception as clf_error:
          print(f"Error in makeNClassifiers: {clf_error}")
          continue

      else:
        clf = base_estimator
        best_params = None
        best_score = None
        train_test_delta = 0.0000
        mean_test_score = 0.0000
        std_test_score = 0.0000
        # print(f"Warning: {selected_method} not recognized, fitting default {model_name_str}")

      clf.fit(X, y)

      end_time = time.time()  # End timer
      elapsed_time = end_time - start_time

      if clf != base_estimator: 
        
        # Pack all objects from clf into dict "results"
        best_params = clf.best_params_
        best_score = clf.best_score_

        mean_test_score = clf.cv_results_['mean_test_score'][clf.best_index_]
        std_test_score  = clf.cv_results_['std_test_score'][clf.best_index_]
        pessimistic_test_score = mean_test_score - num_of_STD * std_test_score

        mean_train_score  = clf.cv_results_['mean_train_score'][clf.best_index_]

        train_test_delta = mean_train_score - mean_test_score

        print(f"{clf.cv_results_['params'][clf.best_index_]} gives the best worst-case test result: (mean - {num_of_STD}*std): {pessimistic_test_score}")
        print(f"Best model found and fitted in {elapsed_time:.4f} seconds")
        print(f"\n") 

      results.append( {
                        'model_name':       model_name_str,
                        'optimalizer':      method,

                        'classifier':       clf,
                        'best_score':       best_score,
                        'best_params':      best_params,      
                        'train_test_delta': train_test_delta, # Higher value -> more overfitted
                        'mean_test_score':  mean_test_score,
                        'std_test_score':   std_test_score,
                      })
      
  return results

def evaluateCLFs(results:           List[Dict[str, Any]],
                 test_df:           pd.DataFrame,
                 test_labels:       Sequence,
                 clf_results_path:  str
                ) -> Tuple[Dict[str, Any], List[float]]:
  
  '''
  Evaluates a list of trained classifiers on a test dataset.

  This function iterates through a list containing results for multiple
  classifiers (typically generated from hyperparameter tuning like
  GridSearchCV or RandomizedSearchCV). For each classifier, it calculates
  performance metrics (Accuracy, F1-score) on the provided test data and
  prints these along with validation metrics passed in the `results`.

  It identifies the best classifier based on the highest F1-score achieved
  on the test set. It also calculates and prints the accuracy of a baseline
  DummyClassifier ('most_frequent' strategy).

  Finally, it saves the input `results` list, the dictionary corresponding
  to the best identified classifier (`best_result`), and a list of the
  calculated test accuracies to a specified file using joblib.
  '''


  # --- 1. Show baseline accuracy ---
  accuracy_list = []
  highest_score = 0.0

  dummy_clf = dummy.DummyClassifier(strategy="most_frequent")
  dummy_clf.fit(test_df, test_labels)
  dummy_clf.predict(test_df)
  dummy_score = dummy_clf.score(test_df, test_labels)

  print("Dummy Classifier accuracy:", round(dummy_score, 4))
  print()
  

  # --- 2. Print metrics and find best result ---
  for result_dict in results:
    
    model_name        = result_dict['model_name']
    clf               = result_dict['classifier']
    mean_test_score   = result_dict['mean_test_score']
    std_test_score    = result_dict['std_test_score']
    train_test_delta  = result_dict['train_test_delta']
    optimalizer       = result_dict['optimalizer']

    test_predict      = clf.predict(test_df)

    accuracy_score    = metrics.accuracy_score(test_labels, test_predict)
    f1_score          = metrics.f1_score(test_labels, test_predict, average="weighted")
    # recall_score      = metrics.recall_score(test_labels, test_predict, average="weighted")
    # precision_score   = metrics.precision_score(test_labels, test_predict, average="weighted")

    print(f"{model_name}: {optimalizer}")
    print(f"Accuracy: \t {accuracy_score:.4f}")
    print(f"f1_score: \t {f1_score:.4f}")
    print(f"Valid. mean: \t {mean_test_score:.4f}")
    print(f"Valid. std: \t {std_test_score:.4f}")
    print(f"Valid. delta: \t {train_test_delta:.4f}")
    print("-" * 23)

    if f1_score > highest_score:
      highest_score     = f1_score
      best_model        = model_name
      best_optimalizer  = optimalizer
      best_result       = result_dict

    accuracy_list.append(accuracy_score)

  print(f"Best clf: \t {best_model}: {best_optimalizer}")
  print(f"f1_score: \t {highest_score:.4f}")
  print("")


  # --- 3. Save results in a file --- 
  try:
    # Bundle classifier-related results
    results_to_save = {
        'n_results': results,
        'result': best_result,
        'accuracy_list': accuracy_list
    }
    
    joblib.dump(results_to_save, clf_results_path)
    print(f"Saved classifier results to {clf_results_path}")

  except Exception as e:
    print(f"Error in saving classifier results: {e}") 

  return best_result, accuracy_list