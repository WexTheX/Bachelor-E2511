import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from SignalProcessing.get_Freq_Domain_features_of_signal import getFFT, getWelch
from sklearn import svm, metrics, dummy
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.decomposition import PCA


# Plot of normal distribution, WIP
def normDistPlot(dataset, size):
  mean = dataset["mean_accel_X"]
  sd = dataset["sd_accel_X"]
    
  values = np.random.normal(mean, sd, size)

  plt.hist(values, 100)
  plt.axvline(values.mean(), color='k', linestyle='dashed', linewidth=2)
  plt.figure()

# Plot of FFT of sets and variables, REDO needed
def plotFFT(sets, variables):
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

# Plot of Welch Method
def plotWelch(signal, feature, fs, filtering=True, omega_n = 15, order = 3):
  df = pd.read_csv(signal+".csv")
  x = df[feature]
  freq, psd = getWelch(x, fs, filtering, omega_n, order)

  plt.semilogy(freq, psd)  # Log scale for better visibility

def testWelch(sets_n, variables_n, fs):
  omega_range = [10, 20, 30, 40, 50]
  filter_order = [1, 2, 3, 4, 5, 6, 7, 8, 9]

  freq, psd = getWelch(sets_n, variables_n, fs, filterOn = True)
  
def biplot(X, trainLabels, PCATest, n_components, separate_types, models, optimization_methods, titles, accuracy_list):
  
  # PCA_object = PCA(n_components = n_components)

  if n_components == 2:

    coeff = PCATest.components_.T
    labels = PCATest.feature_names_in_

    # loadings = PCATest.components_.T * np.sqrt(PCATest.explained_variance_)
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(loadings, annot=True, cmap='coolwarm', xticklabels=['PC1', 'PC2'], yticklabels=PCATest.feature_names_in_)
    # plt.title('Feature Importance in Principal Components')
    
    xs, ys = X[0], X[1]

    plt.figure(figsize=(10, 8))

    if(separate_types):
      label_mapping = {'IDLE': (0.0, 0.0, 0.0)  , 
                       'GRINDBIG': (1.0, 0.0, 0.0),'GRINDMED': (1.0, 0.5, 0.0), 'GRINDSMALL': (1.0, 0.0, 0.5),
                       'SANDSIM': (0.0, 1.0, 0.0), 
                       'WELDALTIG': (0.0, 0.0, 1.0), 'WELDSTMAG': (0.5, 0.0, 1.0), 'WELDSTTIG': (0.0, 0.5, 1.0)}
    else:
      label_mapping = {'IDLE': (0.0, 0.0, 0.0)  , 'GRINDING': (1.0, 0.0, 0.0), 'SANDSIMULATED': (0.0, 1.0, 0.0), 'WELDING': (0.0, 0.0, 1.0)}

    y_labels = np.array(trainLabels)
    mappedLabels = np.array([label_mapping[label] for label in trainLabels])

    plt.scatter(xs, ys, c=mappedLabels#, cmap='viridis'
                )
    
    
    # Check if there is actually 2 components in the clf
    if models[0].n_features_in_ == 2:

      fig, sub = plt.subplots(2, 2)
      plt.subplots_adjust(wspace=0.4, hspace=0.4)

      for clf, method, title, accuracy, ax in zip(models, optimization_methods, titles, accuracy_list, sub.flatten()):
          
          disp = DecisionBoundaryDisplay.from_estimator(
              clf,
              X,
              response_method="predict",
              cmap=plt.cm.coolwarm,
              alpha=0.8,
              ax=ax,
              xlabel='PC1',
              ylabel='PC2',
          )
          ax.scatter(xs, ys, c=mappedLabels, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
          ax.set_xticks(())
          ax.set_yticks(())
          ax.set_title(str(method) + "\n" + "Accuracy: " + str(accuracy) + "\n" + str(title) )


    # Uncomment if you want arrows
    # for i in range(len(coeff)):
    #     plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='r', alpha=0.5)
    #     plt.text(coeff[i, 0] * 1.2, coeff[i, 1] * 1.2, labels[i], color='g')

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    # plt.title("Biplot")
    # plt.figure()

  elif n_components == 3:

    coeff = PCATest.components_.T
    labels = PCATest.feature_names_in_

    # loadings = PCATest.components_.T * np.sqrt(PCATest.explained_variance_)
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(loadings, annot=True, cmap='coolwarm', xticklabels=['PC1', 'PC2', 'PC3'], yticklabels=PCATest.feature_names_in_)
    # plt.title('Feature Importance in Principal Components')

    xs, ys, zs = X[0], X[1], X[2]

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    if (separate_types):
      label_mapping = {'IDLE': (0.0, 0.0, 0.0)  , 
                       'GRINDBIG': (1.0, 0.0, 0.0),'GRINDMED': (1.0, 0.5, 0.0), 'GRINDSMALL': (1.0, 0.0, 0.5),
                       'SANDSIM': (0.0, 1.0, 0.0), 
                       'WELDALTIG': (0.0, 0.0, 1.0), 'WELDSTMAG': (0.5, 0.0, 1.0), 'WELDSTTIG': (0.0, 0.5, 1.0)}
    else:
      label_mapping = {'IDLE': (0.0, 0.0, 0.0)  , 'GRINDING': (1.0, 0.0, 0.0), 'SANDSIMULATED': (0.0, 1.0, 0.0), 'WELDING': (0.0, 0.0, 1.0)}
    y_labels = np.array(trainLabels)
    mappedLabels = np.array([label_mapping[label] for label in trainLabels])

    # Create 3D scatter plot
    sc = ax.scatter(xs, ys, zs, c=mappedLabels#, cmap='inferno'
                    )

    # Draw arrows for the components
    # for i in range(len(coeff)):
    #     ax.quiver(0, 0, 0, coeff[i, 0], coeff[i, 1], coeff[i, 2], color='r', alpha=0.5)

    #     ax.text(coeff[i, 0] * 1.2, coeff[i, 1] * 1.2, coeff[i, 2] * 1.2, labels[i], color='g')

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title("3D Biplot")
    # plt.figure()

  elif 3 < n_components < 10:

    coeff = PCATest.components_.T
    labels = PCATest.feature_names_in_

    loadings = PCATest.components_.T * np.sqrt(PCATest.explained_variance_)

    loadingsPerc = (loadings - np.min(loadings)) / (np.max(loadings) - np.min(loadings))
    plt.figure(figsize=(10, 8))
    sns.heatmap(loadingsPerc, annot=True, cmap='coolwarm', xticklabels=['PC1', 'PC2'], yticklabels=PCATest.feature_names_in_)
    plt.title('Feature Importance in Principal Components')

  else:
    print(f"Too many principal components to plot in a meaningful way")
    pass


def PCA_table_plot(X, n_components):
  
  if 3 < n_components < 10:

    PCA_object = PCA(n_components = n_components)
    PCA_object.fit(X)
    
    loadings = PCA_object.components_.T * np.sqrt(PCA_object.explained_variance_)
    loadings_percantage = (loadings - np.min(loadings)) / (np.max(loadings) - np.min(loadings))
    print(loadings_percantage)

    print(f"Total amount of features: {len(X.columns)}")

    for i in range(len(X.columns) // 34):
      

      loadings_percantage_part = loadings_percantage[i*34:(i*34+34)]
      feature_names_part = PCA_object.feature_names_in_[i*34:(i*34+34)]

      plt.figure(figsize=(10, 8))
      sns.heatmap(loadings_percantage_part, annot=True, cmap='coolwarm', xticklabels=['PC1', 'PC2'], yticklabels = feature_names_part)
      plt.title('Feature Importance in Principal Components')

  else:
    print(f"Too many principal components to plot in a meaningful way")
    pass


def new_biplot(train_data_scaled, train_labels, separate_types):
  
  PCA_object = PCA(n_components = 2)
  X = pd.DataFrame(PCA_object.fit_transform(train_data_scaled))
  
  # This function will always plot the 2 most important PC's
  coeff = PCA_object.components_.T
  labels = PCA_object.feature_names_in_

  # loadings = PCATest.components_.T * np.sqrt(PCATest.explained_variance_)
  # plt.figure(figsize=(10, 8))
  # sns.heatmap(loadings, annot=True, cmap='coolwarm', xticklabels=['PC1', 'PC2'], yticklabels=PCATest.feature_names_in_)
  # plt.title('Feature Importance in Principal Components')
  
  xs, ys = X[0], X[1]

  plt.figure(figsize=(10, 8))

  if(separate_types):
    label_mapping = {'IDLE': (0.0, 0.0, 0.0)  , 
                    'GRINDBIG': (1.0, 0.0, 0.0),'GRINDMED': (1.0, 0.5, 0.0), 'GRINDSMALL': (1.0, 0.0, 0.5),
                    'SANDSIM': (0.0, 1.0, 0.0), 
                    'WELDALTIG': (0.0, 0.0, 1.0), 'WELDSTMAG': (0.5, 0.0, 1.0), 'WELDSTTIG': (0.0, 0.5, 1.0)}
  else:
    label_mapping = {'IDLE': (0.0, 0.0, 0.0)  , 'GRINDING': (1.0, 0.0, 0.0), 'SANDSIMULATED': (0.0, 1.0, 0.0), 'WELDING': (0.0, 0.0, 1.0)}

  # y_labels = np.array(train_labels)
  mappedLabels = np.array([label_mapping[label] for label in train_labels])

  plt.scatter(xs, ys, c=mappedLabels#, cmap='viridis'
              )



def plot_SVM_boundaries(X, train_labels, separate_types,
                        models, optimization_methods, titles, accuracy_list):

  # Check if there is actually 2 components in the clf
  if models[0].n_features_in_ == 2:

    if(separate_types):
      label_mapping = {'IDLE': (0.0, 0.0, 0.0), 
                      'GRINDBIG': (1.0, 0.0, 0.0), 'GRINDMED': (1.0, 0.5, 0.0), 'GRINDSMALL': (1.0, 0.0, 0.5),
                      'SANDSIM': (0.0, 1.0, 0.0), 
                      'WELDALTIG': (0.0, 0.0, 1.0), 'WELDSTMAG': (0.5, 0.0, 1.0), 'WELDSTTIG': (0.0, 0.5, 1.0)}
    else:
      label_mapping = {'IDLE': (0.0, 0.0, 0.0), 'GRINDING': (1.0, 0.0, 0.0), 'SANDSIMULATED': (0.0, 1.0, 0.0), 'WELDING': (0.0, 0.0, 1.0)}

    mapped_labels = np.array([label_mapping[label] for label in train_labels])



    xs, ys = X[0], X[1]
    
    fig, sub = plt.subplots(2, 2)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    for clf, method, title, accuracy, ax in zip(models, optimization_methods, titles, accuracy_list, sub.flatten()):
        
        disp = DecisionBoundaryDisplay.from_estimator(
            clf,
            X,
            response_method="predict",
            cmap=plt.cm.coolwarm,
            alpha=0.8,
            ax=ax,
            xlabel='PC1',
            ylabel='PC2',
        )
        ax.scatter(xs, ys, c=mapped_labels, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(str(method) + "\n" + "Accuracy: " + str(accuracy) + "\n" + str(title) )

  else:
    print(f"Classifiers has {models[0].n_features_in_} features, need 2 to plot SVM boundaries")
