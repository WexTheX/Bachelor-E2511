import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

from sklearn import metrics
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.decomposition import PCA
from sklearn.model_selection import LearningCurveDisplay, ShuffleSplit, StratifiedKFold
from typing import List, Dict, Any, Tuple, Sequence, Optional
from matplotlib.lines import Line2D
from matplotlib.figure import Figure
from collections import defaultdict
from collections import Counter

from SignalProcessing.get_Freq_Domain_features_of_signal import getFFT, getWelch

def normDistPlot(dataset: Any,
                 size:    Any
                 ) -> None:
  
  '''Plots a histogram of randomly generated normally distributed data.

  Uses the 'mean_accel_X' and 'sd_accel_X' values from the input dataset
  to generate 'size' random samples from a normal distribution. It then
  plots a histogram of these samples and adds a vertical dashed line
  indicating the mean of the generated samples.'''

  mean = dataset["mean_accel_X"]
  sd = dataset["sd_accel_X"]
    
  values = np.random.normal(mean, sd, size)

  plt.hist(values, 100)
  plt.axvline(values.mean(), color='k', linestyle='dashed', linewidth=2)
  plt.figure()

  return None

def plotFFT(sets:       Any,
            variables:  Any
            ) -> None:

  '''Plots the FFT magnitude spectrum for specified variables across datasets.

  Iterates through each dataset identifier in 'sets' and each variable
  name in 'variables'. For each combination, it retrieves the FFT results
  (frequency axis, FFT components, and size) using the `getFFT` function.
  It then plots the scaled magnitude of the positive frequency components
  (2.0/N * |Y(f)|) onto the *same* matplotlib axes. Finally, it adds a
  legend identifying each plotted line.'''

  #TODO: Redo this function

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

  return None

def plotWelch(signal_path:  str,
              feature:      str,
              fs:           int,
              filtering:    bool=True,
              omega_n:      int = 15,
              order:        int = 3
              ) -> None:
  
  '''Computes and plots the Power Spectral Density (PSD) using Welch's method.

  Reads data from a CSV file named '<signal>.csv', extracts the specified
  'feature' column. It then computes the PSD using the `getWelch` function,
  optionally applying a filter beforehand based on the 'filtering' flag
  and related parameters ('omega_n', 'order'). The resulting PSD is plotted
  against frequency on a semi-logarithmic scale.'''
  
  # Plot of Welch Method
  df = pd.read_csv(signal_path+".csv")
  x = df[feature]
  freq, psd = getWelch(x, fs, filtering, omega_n, order)

  plt.semilogy(freq, psd)  # Log scale for better visibility

  return None

def testWelch(sets_n:       Any,
              variables_n:  Any,
              fs:           int
              ) -> None:
  
  '''Attempts to test the Welch method calculation (currently incomplete/incorrect).

  This function defines ranges for filter parameters but does not use them.
  It calls the `getWelch` function with inputs `sets_n` and `variables_n`
  which might be incorrect if `getWelch` expects a 1D numerical array as
  its primary data input (like in `plotWelch`). The calculated frequency
  and PSD results are currently not used (e.g., plotted or returned).'''

  omega_range = [10, 20, 30, 40, 50]
  filter_order = [1, 2, 3, 4, 5, 6, 7, 8, 9]

  freq, psd = getWelch(sets_n, variables_n, fs, filterOn = True)

  return None

def plotPCATable(PCA_object:          Any,
                 features_per_table:  int,
                 lower_pc_limit:      int = 2,
                 upper_pc_limit:      int = 10
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
  if lower_pc_limit <= n_components <= upper_pc_limit:
    
    loadings = PCA_object.components_.T * np.sqrt(PCA_object.explained_variance_)
    loadings_percantage = (loadings - np.min(loadings)) / (np.max(loadings) - np.min(loadings))

    # print(f"Total amount of features: {len(loadings)}")
    
    for i in range(len(loadings) // features_per_table):
      
      start_idx = i * features_per_table
      end_idx   = start_idx + features_per_table

      loadings_percantage_part  = loadings_percantage[start_idx:end_idx]
      feature_names_part        = PCA_object.feature_names_in_[start_idx:end_idx]
      
      fig, ax = plt.subplots(figsize=(10, 8))

      # plt.figure(figsize=(10, 8))
      sns.heatmap(loadings_percantage_part, annot=True, cmap='coolwarm', xticklabels=['PC1', 'PC2'], yticklabels = feature_names_part, ax=ax)
      plt.title('Feature Importance in Principal Components')

      figures_list.append(fig)
  
  else:
    print(f"Too many principal components ({n_components}) to plot a PCA table in a meaningful way.")
    return None
  
  return figures_list

def biplot(feature_df:              pd.DataFrame,
           scaler:                  Any,
           train_labels:            Sequence,
           label_mapping:           Dict[str, Any],
           window_length_seconds:   int,
           want_arrows:             bool = False,
           feature_start_index:     int = 90,
           num_features_per_sensor: int = 15,
           output_filename:         str = "plots/biplot.png"
           ) -> Figure:
  
  '''
  Generates a 2D scatter plot of data projected onto its first two Principal Components (PCs).

  Performs PCA on the scaled input data to reduce it to 2 dimensions.
  It then creates a scatter plot where each point represents a sample projected
  onto the first two PCs (PC1 vs PC2). Points are colored according to their
  original labels using the provided `label_mapping`.
  '''


  # --- 1. Fit scaler and PCA transform ---
  try:
    PCA_object        = PCA(n_components = 2)
    total_data_scaled = scaler.fit_transform(feature_df)
    X                 = pd.DataFrame(PCA_object.fit_transform(total_data_scaled))

  except Exception as e:
    print(f"Error during scaling or PCA in biplot: {e}")
    raise

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
    coeff = PCA_object.components_.T[feature_start_index:feature_start_index + num_features_per_sensor]
    
    for i in range(len(coeff)):
        plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='r', alpha=0.5)
        plt.text(coeff[i, 0] * 1.2, coeff[i, 1] * 1.2, total_data_scaled.columns[i+feature_start_index], color='g')

  plt.xlabel("PC1")
  plt.ylabel("PC2")
  plt.title(f"Complete dataset in 2 Principal Components, {window_length_seconds}s windows")

  plt.legend(handles=legend_handles, title="Labels", loc='best') # 'best' tries to find optimal location


  # --- 3. Save plot ---
  try:
    fig.savefig(output_filename, dpi=300, bbox_inches='tight')
  except Exception as e:
    print(f"Error saving plot to {output_filename}: {e}")  

  return fig

def biplot3D(feature_df:              pd.DataFrame,
             scaler:                  Any,
             train_labels:            Sequence,
             label_mapping:           Dict[str, Any],
             window_length_seconds:   int,
             want_arrows:             bool = False,
             feature_start_index:     int = 90,
             num_features_per_sensor: int = 15,
             output_filename:         str = "plots/biplot3D.png"
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
  plt.title(f"Complete dataset in 3 Principal Components, {window_length_seconds}s windows")

  plt.legend(handles=legend_handles, title="Labels", loc='best') # 'best' tries to find optimal location


  # --- 3. Save plot ---
  try:
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
  except Exception as e:
    print(f"Error saving plot to {output_filename}: {e}")  

  return fig

def plotDecisionBoundaries(X:               pd.DataFrame, 
                           train_labels:    Sequence, 
                           label_mapping:   Dict[str, Any],
                           results:         List[Dict[str, Any]],
                           accuracy_list:   List[float],
                           cmap:            str,
                           output_filename: str = "plots/decision_boundaries.png"
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
                grid_resolution=1000,
                response_method="predict",
                plot_method='pcolormesh',
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

    for j in range(num_plots, len(axes)):
      fig.delaxes(axes[j])

    fig.tight_layout()

  
    # --- 2. Save plot ---
    try:
      fig.savefig(output_filename, dpi=300, bbox_inches='tight')
    except Exception as e:
      print(f"Error saving plot to {output_filename}: {e}")  

    return fig

  else:
    print(f"Warning: Cannot plot decision boundaries. Classifiers has {X.shape[1]} features, must be 2.")
    return None

def confusionMatrix(labels:           Sequence,
                    X_test:           pd.DataFrame, 
                    activities:       Sequence, 
                    result:           Dict[str, Any],
                    output_filename:  str = "plots/Confusion_matrix"
                    ) -> Figure:
  
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

  fig = plt.figure(figsize=(10, 8))

  try:
    fig, ax = plt.subplots(figsize=(10, 8))

    test_predict = clf.predict(X_test)
    
    conf_matrix = metrics.confusion_matrix(labels, test_predict, labels=activities, normalize="true")
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', xticklabels=activities, yticklabels=activities, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f'Confusion matrix, {model}: {optimalizer}')

  except Exception as e:
    print(f"Unable to plot confusion matrix: {e}")

  # --- 3. Save plot ---
  try:
    fig.savefig(output_filename, dpi=300, bbox_inches='tight')
  except Exception as e:
    print(f"Error saving plot to {output_filename}: {e}")

  return fig

def plotFeatureImportance(pca:                    Any,
                          original_feature_names: List[str],
                          threshold:              Optional[float] = None,
                          percentile_cutoff:      float = 25.0,
                          output_filename:        str = "plots/feature_importance.png"
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

  feature_dict = {}

  # --- 1. Calculate Importance Vector and threshold ---
  try:  
    # Absolute components weighted by explained variance
    vector = (np.abs(pca.components_.T) @ (pca.explained_variance_ratio_))
    normalized_vector_percentage = 100 * (vector / vector.sum())
    
    if threshold is None:
      threshold = np.percentile(normalized_vector_percentage, percentile_cutoff)

  except AttributeError as e:
    print(f"Error: PCA object passed to evaluateFeatureImportance does not have correct attributes: {e}")
    raise

  sorted_vector = np.sort(vector)[::-1]
  sorted_normalized_vector_percentage = 100 * (sorted_vector / sorted_vector.sum())

  
  # --- 2. Identify and Format Least Important Features ---
  # Make a dict with {feature_name: importance_value}
  try:
    feature_dict = {name: normalized_vector_percentage[i] for i, name in enumerate(original_feature_names)}
  except Exception as e:
    error_message = (
      f" - Number of importance scores calculated: {len(normalized_vector_percentage)}"
      f" - Number of feature names provided: {len(original_feature_names)}\n"
      f" - This suggests feature extraction must be done again, or norm_IMU changed.\n")
    print(error_message)


  sensor_importance_df, suffix_importance_df, fig_1 = getSensorAndSuffixImportance(feature_dict)

  low_value_feature_dict = {key: value for key, value in feature_dict.items() if value < threshold}
  sorted_dict_items = sorted(low_value_feature_dict.items(), key=lambda item: item[1], reverse=True)
  dict_string = "Least important features:\n\n" + "\n".join([f"{key}: {value:.3f}" for key, value in sorted_dict_items])


  # --- 3. Plotting ---
  
  if sorted_dict_items:

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

    ax.set_title(f"Importance of {len(original_feature_names)} original features to {pca.n_components} Principal Components", fontsize=16)
    ax.set_xlabel("Sorted features", fontsize=12)
    ax.set_ylabel("% Contribution to all PC's", fontsize=12)

    ax.grid(True, linestyle=':',  alpha=0.6)
    ax.set_xticks(np.arange(0, len(sorted_vector), 10))
    ax.tick_params(axis='x', rotation=45)

    plt.subplots_adjust(right=0.73)

  if not sorted_dict_items:
    print(f"No features below {threshold} in importance.")


  # --- 4. Save plot ---
  try:
    fig.savefig(output_filename, dpi=300, bbox_inches='tight')
  except Exception as e:
        print(f"Error saving plot to {output_filename}: {e}")

  fig_list = [fig, fig_1]
  
  return fig_list

def getSensorAndSuffixImportance(feature_dict:    dict[Any, Any],
                                 output_filename: str = "plots/sensor_and_suffix_importance.png"
                                 ) -> Tuple[pd.DataFrame, pd.DataFrame, Figure]:

  '''
  Calculates and plots aggregated feature importance grouped by sensor and suffix.

  This function takes a dictionary or Pandas Series mapping feature names to
  their importance scores. It assumes feature names follow the format
  "suffix_sensor" (e.g., "mean_AccX", "std_GyroY"). It aggregates the
  importance scores separately for each unique sensor identifier and each
  unique suffix (statistical feature type). Finally, it generates and saves
  a bar plot visualizing these aggregated importances and returns the
  aggregated data as Pandas DataFrames.
  '''


  # --- 1. Calculate importance ---
  sensor_importance = defaultdict(float)
  suffix_importance = defaultdict(float)

  for feature_name, importance in feature_dict.items():
    parts = feature_name.split('_', 1)
    suffix, sensor = parts

    sensor_importance[sensor] += importance
    suffix_importance[suffix] += importance

  sensor_df = pd.DataFrame(sensor_importance.values(), index=sensor_importance.keys(), columns=['Importance']).sort_values(by='Importance', ascending=False)
  suffix_df = pd.DataFrame(suffix_importance.values(), index=suffix_importance.keys(), columns=['Importance']).sort_values(by='Importance', ascending=False)


  # --- 2. Plotting ---
  fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

  # Plot Sensor Importance
  sensor_df.plot(kind='bar', ax=axes[0], legend=False, color='skyblue')
  axes[0].set_title('Feature Importance by Sensor')
  axes[0].set_ylabel('Total Importance [%]')
  axes[0].set_xlabel('Sensor Name')
  axes[0].tick_params(axis='x', rotation=45)

  # Plot Suffix Importance
  suffix_df.plot(kind='bar', ax=axes[1], legend=False, color='lightcoral')
  axes[1].set_title('Feature Importance by Suffix (Statistic Type)')
  axes[1].set_ylabel('Total Importance [%]')
  axes[1].set_xlabel('Statistical feature')
  axes[1].tick_params(axis='x', rotation=45)

  plt.tight_layout()


  # --- 3. Save plot ---
  try:
    fig.savefig(output_filename, dpi=300, bbox_inches='tight')
  except Exception as e:
    print(f"Error saving plot to {output_filename}: {e}")

  return sensor_df, suffix_df, fig

def screePlot(pca:              Any,
              output_filename:  str = "plots/PCA_scree_plot.png"
              ) -> Figure:

  '''
  Generates and saves a scree plot from a fitted PCA object.

  This function takes a fitted PCA result and creates a scree plot, which
  visualizes the proportion of variance explained by each principal component.
  The plot helps in determining the 'elbow point' to select the optimal
  number of components. The generated plot is also saved to a file.
  '''

  fig = plt.figure(figsize=(8, 5))

  try:
    n_components = pca.n_components_
    explained_variance_ratio = pca.explained_variance_ratio_

    # --- 1. Create the Scree Plot ---
    component_numbers = np.arange(n_components) + 1

    # Plot individual component variance
    plt.plot(component_numbers, explained_variance_ratio, 'o-', linewidth=2, label='Individual Variance')

    # Add labels and title
    plt.title('Scree Plot')
    plt.xlabel('Principal Component Number')
    plt.ylabel('Proportion of Variance Explained')
    plt.xticks(component_numbers, rotation=90)
    plt.legend()
    plt.grid(True)

  except Exception as e:
    print(f"PCA error in screePlot: {e}")


  # --- 2. Save file ---
  try:
    fig.savefig(output_filename, dpi=300, bbox_inches='tight')
  except Exception as e:
    print(f"Error saving plot to {output_filename}: {e}")

  return fig

def plotLearningCurve(results:          List[Dict[str, Any]],
                      X:                pd.DataFrame,
                      y:                Sequence,
                      cv_string:        str = "SKF", #SKF = StratifiedKFold, SS = ShuffleSplit
                      n_splits:         int = 5,
                      train_sizes:      np.ndarray = np.linspace(0.1, 1.0, 15),
                      output_filename:  str = "plots/Learning_curve.png",
                      ) -> Figure:
  
  '''
  Generates and saves learning curve plots for multiple classifiers.

  Takes a list of dictionaries, each containing a scikit-learn classifier,
  and plots their learning curves on a grid using LearningCurveDisplay.
  The plots show training and cross-validation (test) scores against
  varying training set sizes.
  '''


  # --- 1. Plotting Setup ---

  if cv_string == "SKF":
    cv_type = StratifiedKFold(n_splits=n_splits)
  elif cv_string == "SS":
    cv_type = ShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=0)

  models = []

  for result in results:
    models.append(result['classifier'])

  # n_models = len(models)
  # n_rows = (n_models + n_cols - 1) // n_cols

  # fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(5 * n_cols, 4 * n_rows), sharey=True)
  # axes = axes.flatten()  # Flatten in case it's 2D array of axes

  num_plots   = len(models)
  ncols       = math.ceil(math.sqrt(num_plots))
  nrows       = math.ceil(num_plots / ncols)

  fig_width   = ncols * 4.5
  fig_height  = nrows * 4

  fig, axes   = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), squeeze=False)
  axes        = axes.flatten()  # Flatten in case it's 2D array of axes

  # --- 2. CV  ---
  common_params = {
      "X": X,
      "y": y,
      "train_sizes": train_sizes,
      "cv": cv_type,
      "score_type": "both",
      "n_jobs": -1,
      "line_kw": {"marker": "o"},
      "std_display_style": "fill_between",
      "score_name": "Accuracy",
  }


  # --- 3. Generate Plots ---
  for ax_idx, estimator in enumerate(models):

    try:
      LearningCurveDisplay.from_estimator(estimator, **common_params, ax=axes[ax_idx], scoring="balanced_accuracy")
      handles, _ = axes[ax_idx].get_legend_handles_labels()
      axes[ax_idx].legend(handles[:2], ["Training Score", "Test Score"])
      axes[ax_idx].set_title(f"Learning Curve for {estimator.__class__.__name__}")

    except Exception as e:
      print(f"Error plotting learning curve: {e}")
      axes[ax_idx].set_title(f"Error plotting {estimator.__class__.__name__}")
      axes[ax_idx].text(0.5, 0.5, "Plotting failed", ha='center', va='center', color='red')

  for j in range(num_plots, len(axes)):
    fig.delaxes(axes[j])

  plt.tight_layout()


  # --- 4. Save file ---
  try:
    fig.savefig(output_filename, dpi=300, bbox_inches='tight')
  except Exception as e:
    print(f"Error saving plot to {output_filename}: {e}")

  return fig


def datasetOverview(labels:                 Sequence,
                    window_length_seconds:  int,
                    test_size:              float,
                    output_filename:        str = "plots/distribution_of_labels"
                    ) -> Figure:
  
  train_size = 1-test_size

  counts_overview = {}

  counts = Counter(labels)
  total = sum(counts.values())

  for k, v in counts.items():
    fraction = round((v / total), 3)
    counts_overview[k] = v, fraction

  df = pd.DataFrame([counts_overview])

  sorted_counts = dict(sorted(counts.items()))
  plot_labels = list(sorted_counts.keys())
  plot_values = list(sorted_counts.values())

  # Create Figure and Axes objects
  fig, ax = plt.subplots(figsize=(10, 6)) # Adjust figsize as needed

  bars = ax.bar(plot_labels, plot_values, color='skyblue')
  
  ax.set_xlabel("Class")
  ax.set_ylabel("Number of windows")
  ax.set_title(f"Distribution of classes in the dataset, {window_length_seconds} second windows")
  plt.xticks(rotation=45, ha="right") # Rotate x-axis labels if they overlap
  ax.grid(axis='y', linestyle='--')
  # ax.set_ylim(0, (max(plot_values) + 50))

  total_minutes=0.0
  total_samples =0
  # Optional: Add text labels on top of bars
  for bar in bars:
      
      yval = bar.get_height()

      ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.005 * max(plot_values), # Adjust offset
              int(yval), # Display integer count
              ha='center', va='bottom')
      
      total_minutes += yval * window_length_seconds / 60
      total_samples += yval

  # Add total_seconds as a box annotation in the plot
  ax.text(0.95, 0.95,
          f"Total minutes: {int(total_minutes):,}\nTotal samples: {int(total_samples):,}\nTrain samples: {int(train_size * total_samples)}",
          transform=ax.transAxes,
          fontsize=12,
          verticalalignment='top',
          horizontalalignment='right',
          bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

  plt.tight_layout()

  try:
    fig.savefig(output_filename, dpi=300, bbox_inches='tight')
  except Exception as e:
    print(f"Error saving plot to {output_filename}: {e}")

  print(df)

def plotScoreVsWindowLength(window_lengths, mean_scores, std_scores, score_type='F1 Score', color='blue', filename=None):
    """
    Plots a score (e.g., F1 score or Accuracy) with ±1 standard deviation shaded area and saves the plot.
    
    Parameters:
        window_lengths (list or array): The x-axis values (window lengths).
        mean_scores (list or array): The mean values of the scores (F1 or Accuracy).
        std_scores (list or array): The standard deviation values for the scores.
        score_type (str): Type of the score ('F1 Score' or 'Accuracy Score'). Default is 'F1 Score'.
        color (str): The color of the plot line and shaded area. Default is 'blue'.
        filename (str): The path to save the plot image. If None, the plot will not be saved.
    """
    np_window_lengths = np.array(window_lengths)
    np_mean_scores = np.array(mean_scores)
    np_std_scores = np.array(std_scores)

    # --- Generalized Plot ---
    plt.figure(figsize=(10, 6))
    plt.plot(np_window_lengths, np_mean_scores, label=f'Mean {score_type}', color=color, marker='o')

    plt.fill_between(np_window_lengths, np_mean_scores - np_std_scores, np_mean_scores + np_std_scores,
                     color=color, alpha=0.2, label=f'Mean ± 1 STD')

    plt.xlabel("Window Length (seconds)")
    plt.ylabel(f"{score_type}")
    plt.title(f"{score_type} vs. Window Length")
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(np.arange(min(np_window_lengths), max(np_window_lengths) + 1, 25))
    plt.ylim(0, 1.05)
    plt.tight_layout()

    # Save the plot if a filename is provided
    if filename:
        try:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved {score_type} plot to {filename}")
        except Exception as e:
            print(f"Error saving {score_type} plot: {e}")

    plt.show()

    return None