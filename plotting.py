import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from SignalProcessing.get_Freq_Domain_features_of_signal import getFFT, getWelch

# Plot of normal distribution, WIP
def normDistPlot(dataset, size):
  mean = dataset["mean_accel_X"]
  sd = dataset["sd_accel_X"]
    
  values = np.random.normal(mean, sd, size)

  plt.hist(values, 100)
  plt.axvline(values.mean(), color='k', linestyle='dashed', linewidth=2)
  plt.figure()

'''
df = pd.read_csv("OutputFiles/features4.csv")
size = 10
print(df)

normDistPlot(df[:1], 800*size)
plt.show()
'''

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
  

# getWelch(file, feature, fs, filterOn)

def biplot_3D(score, trainLabels, PCATest):

    coeff = PCATest.components_.T
    labels = PCATest.feature_names_in_

    loadings = PCATest.components_.T * np.sqrt(PCATest.explained_variance_)
    plt.figure(figsize=(10, 8))
    sns.heatmap(loadings, annot=True, cmap='coolwarm', xticklabels=['PC1', 'PC2', 'PC3'], yticklabels=PCATest.feature_names_in_)
    plt.title('Feature Importance in Principal Components')

    xs = score[0]
    ys = score[1]
    zs = score[2]

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    label_mapping = {'GRIND': 0, 'IDLE': 1, 'WELD': 2}
    y_labels = np.array(trainLabels)
    mappedLabels = np.array([label_mapping[label] for label in trainLabels])

    # Create 3D scatter plot
    sc = ax.scatter(xs, ys, zs, c=mappedLabels, cmap='inferno')

    # Draw arrows for the components
    for i in range(len(coeff)):
        ax.quiver(0, 0, 0, coeff[i, 0], coeff[i, 1], coeff[i, 2], color='r', alpha=0.5)

        ax.text(coeff[i, 0] * 1.2, coeff[i, 1] * 1.2, coeff[i, 2] * 1.2, labels[i], color='g')

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title("3D Biplot")

    plt.show()