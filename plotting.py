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

def PCA_table_plot(X:                 pd.DataFrame, 
                   n_components:      int,
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
  function only generates plots if `n_components` is between 3 and 10
  (inclusive), based on an assumption about visual clarity.
  '''
  
  figures_list: List[Figure] = []

  if 3 <= n_components <= 10:

    PCA_object = PCA(n_components = n_components)
    PCA_object.fit(X)
    
    loadings = PCA_object.components_.T * np.sqrt(PCA_object.explained_variance_)
    loadings_percantage = (loadings - np.min(loadings)) / (np.max(loadings) - np.min(loadings))

    print(f"Total amount of features: {len(X.columns)}")
    
    for i in range(len(X.columns) // features_per_PCA):
      
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
    print(f"Too many principal components to plot in a meaningful way")
  
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

  # Fit scaler and PCA transform
  PCA_object        = PCA(n_components = 2)
  total_data_scaled = scaler.fit_transform(feature_df)
  X                 = pd.DataFrame(PCA_object.fit_transform(total_data_scaled))

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

  return fig

def biplot3D(
    feature_df:     pd.DataFrame,
    scaler:         Any,
    train_labels:   Sequence,
    label_mapping:  Dict[str, Any],
    want_arrows: bool
    ) -> Figure:
  '''
  '''
  # Fit scaler and PCA transform
  PCA_object        = PCA(n_components = 3)
  total_data_scaled = scaler.fit_transform(feature_df)
  X                 = pd.DataFrame(PCA_object.fit_transform(total_data_scaled))

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
  
  if want_arrows:

    # Decide which indices to make arrows from
    start_index = 90
    coeff = PCA_object.components_.T[start_index:start_index+15]
    
    for i in range(len(coeff)):
        plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='r', alpha=0.5)
        plt.text(coeff[i, 0] * 1.2, coeff[i, 1] * 1.2, total_data_scaled.columns[i+start_index], color='g')

  ax.set_xlabel("PC1")
  ax.set_ylabel("PC2")
  ax.set_zlabel("PC3")
  plt.title("Complete dataset in 3 Principal Components")

  plt.legend(handles=legend_handles, title="Labels", loc='best') # 'best' tries to find optimal location

  return fig

def plotBoundaryConditions(X:             pd.DataFrame, 
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
        ax.set_title(str(model_name) + ": " + str(optimalizer) + "\n" + "Accuracy: " + str(accuracy))
        
        # fig.tight_layout()
        return fig

  else:
    print(f"Warning: Cannot plot decision boundaries. Classifiers has {X.shape[1]} features, must be 2.")
    return None
  
def plotKNNboundries(df, clf, labels):
  
    _, ax = plt.subplots()

    disp = DecisionBoundaryDisplay.from_estimator(
    clf,
    df,
    response_method="predict",
    plot_method="pcolormesh",
    shading="auto",
    alpha=0.5,
    ax=ax,
    )
    scatter = disp.ax_.scatter(df.iloc[:, 0], df.iloc[:, 1], c=labels, edgecolors="k")
