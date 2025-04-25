import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

from SignalProcessing.get_Freq_Domain_features_of_signal import getFFT, getWelch
from sklearn import svm, metrics, dummy
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.decomposition import PCA
from typing import List, Dict, Any, Tuple, Sequence, Optional
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure


def normDistPlot(dataset, size):
  # Plot of normal distribution, WIP
  mean = dataset["mean_accel_X"]
  sd = dataset["sd_accel_X"]
    
  values = np.random.normal(mean, sd, size)

  plt.hist(values, 100)
  plt.axvline(values.mean(), color='k', linestyle='dashed', linewidth=2)
  plt.figure()

def plotFFT(sets, variables):
  # Plot of FFT of sets and variables, REDO needed
  label_names = []

  for i in sets:
    for j in variables:
      #tab_txt_to_csv(i+".txt", i+".csv")
      x_yf, x_xf, x_size = getFFT(i, j)
      plt.plot(x_xf, 2.0/x_size*np.abs(x_yf[0:x_size//2]))
      #plt.semilogy(x_xf, 2.0/x_size*np.abs(x_yf[0:x_size//2]))
      label_names.append(i + ", " + j)

      # Plot one plot for each variable, for each set
      # plt.legend(label_names)
      # plt.figure()

    # Plot all variables in same plot, one for each set
    # plt.legend(label_names)
    # plt.figure()
  
  # Plot all sets and variables in same plot
  plt.legend(label_names)
  plt.figure()

def plotWelch(signal, feature, fs, filtering=True, omega_n = 15, order = 3):
  # Plot of Welch Method
  df = pd.read_csv(signal+".csv")
  x = df[feature]
  freq, psd = getWelch(x, fs, filtering, omega_n, order)

  plt.semilogy(freq, psd)  # Log scale for better visibility

def testWelch(sets_n, variables_n, fs):
  omega_range = [10, 20, 30, 40, 50]
  filter_order = [1, 2, 3, 4, 5, 6, 7, 8, 9]

  freq, psd = getWelch(sets_n, variables_n, fs, filterOn = True)

def PCA_table_plot(PCA_object:        Any,
                  #  X:                 pd.DataFrame, 
                  #  n_components:      int,
                   features_per_PCA:  int
                   ) -> Optional[List[Figure]]:
  
  '''
  Generates heatmap(s) visualizing scaled PCA loadings.

  Fits PCA to the input data `X` for the specified `n_components`. It then
  calculates the principal component loadings (components scaled by the
  square root of their explained variance). These loadings are min-max scaled
  across all features and components for visualization purposes.

  If the number of features exceeds `features_per_PCA`, the heatmap is
  split into multiple plots, each displaying a chunk of features. This
  function only generates plots if `n_components` is between 2 and 10
  (inclusive), based on an assumption about visual clarity.
  '''


  # --- 1. Setup ---
  figures_list: List[Figure] = []

  try:
    n_components = PCA_object.n_components_
  except Exception as e:
    print(f"Problem with PCA object in PCA_table_plot: {e}")


  # --- 2. Plot PCA table
  if 2 <= n_components <= 10:
    
    loadings = PCA_object.components_.T * np.sqrt(PCA_object.explained_variance_)
    loadings_percantage = (loadings - np.min(loadings)) / (np.max(loadings) - np.min(loadings))

    # print(f"Total amount of features: {len(loadings)}")
    
    for i in range(len(loadings) // features_per_PCA):
      
      start_idx = i * features_per_PCA
      end_idx   = start_idx + features_per_PCA

      loadings_percantage_part  = loadings_percantage[start_idx:end_idx]
      feature_names_part        = PCA_object.feature_names_in_[start_idx:end_idx]
      
      fig, ax = plt.subplots(figsize=(10, 8))

      # plt.figure(figsize=(10, 8))
      sns.heatmap(loadings_percantage_part, annot=True, cmap='coolwarm', xticklabels=['PC1', 'PC2'], yticklabels = feature_names_part, ax=ax)
      plt.title('Feature Importance in Principal Components')

      figures_list.append(fig)
  
  else:
    print(f"Too many principal components ({n_components}) to plot a PCA table in a meaningful way.")
  
  return figures_list

def biplot(feature_df:        pd.DataFrame,
           scaler:            Any,
           train_labels:      Sequence,
           label_mapping:     Dict[str, Any],
           want_arrows:       bool
           ) -> Figure:
  
  '''
  Generates a 2D scatter plot of data projected onto its first two Principal Components (PCs).

  Performs PCA on the scaled input data to reduce it to 2 dimensions.
  It then creates a scatter plot where each point represents a sample projected
  onto the first two PCs (PC1 vs PC2). Points are colored according to their
  original labels using the provided `label_mapping`.
  '''


  # --- 1. Fit scaler and PCA transform ---
  PCA_object        = PCA(n_components = 2)
  total_data_scaled = scaler.fit_transform(feature_df)
  X                 = pd.DataFrame(PCA_object.fit_transform(total_data_scaled))


  # --- 2. Plotting ---
  xs, ys = X[0], X[1]

  unique_original_labels  = sorted(list(set(train_labels)))
  point_colors            = [label_mapping[label] for label in train_labels]

  # Create legend handles
  legend_handles = []
  for label_name in unique_original_labels:
      
      color   = label_mapping[label_name]
      handle  = Line2D([0], [0], marker='o', color='w', # Dummy data, white line
                      label=label_name, markerfacecolor=color,
                      markersize=8, linestyle='None') # No line connecting markers
      legend_handles.append(handle)

  fig, ax = plt.subplots(figsize=(10, 8))
  ax.scatter(xs, ys, c=point_colors, s=45, edgecolors="k", linewidths=0.35)

  # plt.figure(figsize=(10, 8))
  # plt.scatter(xs, ys, c=point_colors, s=45, edgecolors="k", linewidths=0.35)
  
  if want_arrows:

    # Decide which indices to make arrows from
    start_index = 90
    coeff = PCA_object.components_.T[start_index:start_index+15]
    
    for i in range(len(coeff)):
        plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='r', alpha=0.5)
        plt.text(coeff[i, 0] * 1.2, coeff[i, 1] * 1.2, total_data_scaled.columns[i+start_index], color='g')

  plt.xlabel("PC1")
  plt.ylabel("PC2")
  plt.title("Complete dataset in 2 Principal Components")

  plt.legend(handles=legend_handles, title="Labels", loc='best') # 'best' tries to find optimal location


  # --- 3. Save plot ---
  output_filename = "plots/biplot.png"
  try:
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
  except Exception as e:
    print(f"Error saving plot to {output_filename}: {e}")  

  return fig

def biplot3D(feature_df:      pd.DataFrame,
             scaler:          Any,
             train_labels:    Sequence,
             label_mapping:   Dict[str, Any],
             want_arrows:     bool
             ) -> Figure:
  
  '''
  Generates a 3D scatter plot of data projected onto its first three Principal Components (PCs).

  Performs scaling (using the provided scaler) and then PCA on the input
  feature DataFrame to reduce its dimensionality to 3 components. It then
  creates a 3D scatter plot where each point represents a sample projected
  onto the first three PCs (PC1 vs PC2 vs PC3). Points are colored according
  to their original labels using the provided `label_mapping`.

  Optionally, attempts to draw arrows representing the loadings of a subset
  of the original features onto the first two principal components.
  '''


  # --- 1. Fit PCA object for 3 components and DF ---
  try:
    PCA_object        = PCA(n_components = 3)
    total_data_scaled = scaler.fit_transform(feature_df)
    X                 = pd.DataFrame(PCA_object.fit_transform(total_data_scaled))
  except Exception as e:
    print(f"Error during scaling or PCA in biplot3D: {e}")
    raise

  # --- 2. Plotting ---
  xs, ys, zs = X[0], X[1], X[2]

  unique_original_labels  = sorted(list(set(train_labels)))
  point_colors            = [label_mapping[label] for label in train_labels]

  # Create legend handles
  legend_handles = []
  for label_name in unique_original_labels:
      
      color   = label_mapping[label_name]
      handle  = Line2D([0], [0], marker='o', color='w', # Dummy data, white line
                      label=label_name, markerfacecolor=color,
                      markersize=8, linestyle='None') # No line connecting markers
      legend_handles.append(handle)

  fig = plt.figure(figsize=(10,8))
  ax = fig.add_subplot(111, projection='3d')
  sc = ax.scatter(xs, ys, zs, c=point_colors)
  
  # TODO: Arrows are 2D, doesn't work as intended in 3D
  if want_arrows:  

    # Decide which indices to make arrows from (Hardcoded)
    feature_start_index     = 90 
    num_features_per_sensor = 15
    coeff = PCA_object.components_.T[feature_start_index:feature_start_index + num_features_per_sensor]
    
    # Alternative for all arrows:
    # feature_start_index = 0
    # coeff = PCA_object.components_.T

    for i in range(len(coeff)):
        plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='r', alpha=0.5)
        plt.text(coeff[i, 0] * 1.2, coeff[i, 1] * 1.2, total_data_scaled.columns[i + feature_start_index], color='g')

  ax.set_xlabel("PC1")
  ax.set_ylabel("PC2")
  ax.set_zlabel("PC3")
  plt.title("Complete dataset in 3 Principal Components")

  plt.legend(handles=legend_handles, title="Labels", loc='best') # 'best' tries to find optimal location


  # --- 3. Save plot ---
  output_filename = "plots/biplot3D.png"
  try:
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
  except Exception as e:
    print(f"Error saving plot to {output_filename}: {e}")  

  return fig

def plotDecisionBoundaries(X:             pd.DataFrame, 
                           train_labels:  Sequence, 
                           label_mapping: Dict[str, Any],
                           results:       List[Dict[str, Any]],
                           accuracy_list: List[float],
                           cmap:          str,
                          ) -> Optional[Figure]:
  
  '''
  Plots decision boundaries for multiple classifiers on 2D data.

  Generates a grid of subplots, where each subplot displays the decision
  boundary of a classifier provided in the `results` list. It assumes the
  input data `X` and the fitted classifiers operate on exactly two features.
  A scatter plot of the original data points, colored according to
  `label_mapping`, is overlaid on each decision boundary plot.
  '''

  # --- 1. Plotting ---
  if X.shape[1] == 2:
  
    point_colors = np.array([label_mapping[label] for label in train_labels]) 

    xs, ys = X[0], X[1]

    num_plots   = len(results)
    ncols       = math.ceil(math.sqrt(num_plots))
    nrows       = math.ceil(num_plots / ncols)

    fig_width   = ncols * 4.5
    fig_height  = nrows * 4
    fig, axes   = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), squeeze=False)
    
    for i, (result_dict, accuracy) in enumerate(zip(results, accuracy_list)):
      
      ax = axes.flat[i]
      model_name  = result_dict['model_name']
      clf         = result_dict['classifier']
      optimalizer = result_dict['optimalizer']
      # best_params = result_dict['best_params']

      if clf.n_features_in_ == 2:

        # Background
        disp = DecisionBoundaryDisplay.from_estimator(
                clf,
                X,
                response_method="predict",
                cmap=cmap,
                alpha=0.6,
                ax=ax,
                )
        
        # Foreground
        ax.scatter(xs, ys, c=point_colors, s=25, edgecolors="k", linewidths=0.35)

        # Text
        ax.set_title(f"{model_name}: {optimalizer}", fontsize=10, fontweight='normal')
        ax.text(
          0.98, 0.02,  # X and Y position in axes coords (0=left/bottom, 1=right/top)
          f"{accuracy:.3f}".lstrip("0"),
          transform=ax.transAxes,
          fontsize=8,
          ha='right',
          va='bottom',
          bbox=None
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        # ax.set_aspect('equal', adjustable='box')

      fig.suptitle("Classifier Decision Boundaries", fontsize=14)
      fig.tight_layout()

  
    # --- 2. Save plot ---
    output_filename = "plots/decision_boundaries.png"
    try:
      plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    except Exception as e:
      print(f"Error saving plot to {output_filename}: {e}")  

    return fig

  else:
    print(f"Warning: Cannot plot decision boundaries. Classifiers has {X.shape[1]} features, must be 2.")
    return None

def confusionMatrix(labels:     Sequence,
                    X_test:     pd.DataFrame, 
                    activities: Sequence, 
                    result:     Dict[str, Any]
                    ) -> None:
  
  '''
  Generates and displays a confusion matrix heatmap for classifier predictions.

  Uses a fitted classifier from the `result` dictionary to predict labels for
  the provided test data `X_test`. It then computes the confusion matrix
  comparing these predictions against the true `labels` and visualizes this
  matrix as a heatmap using Seaborn. The plot axes are labeled with the
  provided `activities`, and the title includes the model and optimalizer names.
  '''

  clf         = result['classifier']
  model       = result['model_name']
  optimalizer = result['optimalizer']

  try:
    test_predict = clf.predict(X_test)
    
    conf_matrix = metrics.confusion_matrix(labels, test_predict, labels=activities)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, cmap='coolwarm', xticklabels=activities, yticklabels=activities)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f'Confusion matrix, {model}: {optimalizer}')

  except Exception as e:
    print(f"Unable to plot confusion matrix: {e}")

  return None

def plotFeatureImportance(pca:          Any,
                          threshold:    float = 0.68
                          ) -> Figure:

  '''
  Calculates, analyzes, and visualizes feature importance from PCA results.

  This function computes a feature importance score for each original feature
  based on the absolute values of the principal components weighted by the
  explained variance ratio of each component. It assumes a specific, predefined
  structure for generating the original feature names based on hardcoded lists
  of sensors and feature engineering suffixes (time and frequency domains).

  The function identifies features whose calculated importance falls below the
  specified threshold and lists them in an annotation on a plot. The plot
  displays the sorted importance scores against their rank, includes a
  horizontal line for the threshold, and is saved to the hardcoded path
  'plots/feature_importance.png'.
  '''

  # --- 1. Generate Feature Names (Hardcoded) ---
  original_feature_names  = []
  feature_dict            = {}

  sensors                 = ['accel_X', 'accel_Y', 'accel_Z', 'gyro_X', 'gyro_Y', 'gyro_Z', 'mag_X', 'mag_Y', 'mag_Z', 'temp']
  time_feature_suffixes   = ['mean', 'sd', 'mad', 'max', 'min', 'energy', 'entropy', 'iqr', 'kurtosis', 'skewness', 'correlation']
  freq_sensors            = ['accel_X', 'accel_Y', 'accel_Z', 'gyro_X', 'gyro_Y', 'gyro_Z', 'mag_X', 'mag_Y', 'mag_Z']
  freq_feature_suffixes   = ['psd_mean', 'psd_max', 'psd_min', 'psd_max_freq']

  # Add time features
  for sensor in sensors:
      for suffix in time_feature_suffixes:
          original_feature_names.append(f"{suffix}_{sensor}")

  # Add frequency features
  for sensor in freq_sensors:
      for suffix in freq_feature_suffixes:
          original_feature_names.append(f"{suffix}_{sensor}") 


  # --- 2. Calculate Importance Vector ---
  try:  
    # Absolute components weighted by explained variance
    vector = (np.abs(pca.components_.T) @ (pca.explained_variance_ratio_))
    normalized_vector_percentage = 100 * (vector / vector.sum())

  except AttributeError as e:
    print(f"Error: PCA object passed to evaluateFeatureImportance does not have correct attributes: {e}")
    raise

  sorted_vector = np.sort(vector)[::-1]
  sorted_normalized_vector_percentage = 100 * (sorted_vector / sorted_vector.sum())

  
  # --- 3. Identify and Format Least Important Features ---
  # Make a dict with {feature_name: importance_value}
  feature_dict = {name: normalized_vector_percentage[i] for i, name in enumerate(original_feature_names)}

  low_value_feature_dict = {key: value for key, value in feature_dict.items() if value < threshold}
  sorted_dict_items = sorted(low_value_feature_dict.items(), key=lambda item: item[1], reverse=True)
  dict_string = "Least important features:\n\n" + "\n".join([f"{key}: {value:.3f}" for key, value in sorted_dict_items])
  
  if not sorted_dict_items:
    print(f"No features below {threshold} in importance.")


  # --- 4. Plotting ---
  ranks = np.arange(1, len(sorted_normalized_vector_percentage) + 1)
  fig, ax = plt.subplots(figsize=(8, 5)) # Create figure and axes

  # Plot sorted importance scores
  ax.plot(ranks, sorted_normalized_vector_percentage,
        color='tab:blue',   
        linestyle='-',   
        linewidth=1,        
        marker='o',      
        markersize=4, 
        markeredgecolor='blue',
        label='Sorted feature importance')
    
  # Plot threshold line
  ax.axhline(y=threshold, 
        color='red', 
        linestyle='--',
        linewidth=1.5,  
        label=f'Threshold ({threshold:.3f})')
  
  # Vertical threshold line + text
  intersection_index = np.where(sorted_normalized_vector_percentage <= threshold)[0]
  
  ax.axvline(x=intersection_index[0],
        color='grey',  
        linestyle='--', 
        linewidth=1.5,
        label=f'Threshold Cross ({intersection_index} features above)')
   
  ax.text(intersection_index[0] + 2,  # X: Offset slightly right from the line
          threshold + 0.02,          # Y: Offset slightly above threshold line
          f" Features: {intersection_index[0]} ",
          color='grey',           
          fontsize=8,
          ha='left',         
          va='bottom',
          # bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='grey', alpha=0.8)
           )

  # Add text box with least important features 
  fig.text(0.75, # X position 
         0.85, # Y position 
         dict_string,
         fontsize=7,
         va='top', 
         ha='left', 
         bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.6))

  ax.set_title(f"Feature importance of {pca.n_components} PCs", fontsize=16)
  ax.set_xlabel("Sorted features", fontsize=12)
  ax.set_ylabel("% Contribution to all PC's", fontsize=12)

  ax.grid(True, linestyle=':',  alpha=0.6)
  ax.set_xticks(np.arange(0, len(sorted_vector), 10))
  ax.tick_params(axis='x', rotation=45)

  plt.subplots_adjust(right=0.73)


  # --- 5. Save plot ---
  output_filename = "plots/feature_importance.png"
  try:
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
  except Exception as e:
        print(f"Error saving plot to {output_filename}: {e}")
  
  return Figure