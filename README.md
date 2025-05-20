# Detecting industrial actitivities using machine learning
This project is a collection of code that finds the best hyperparameters and classifiers for detecting industrial activities such as angle grinding and welding.


# What you can do

- Add new data specifically for your use-case to improve accuracy and/or expand on the project
- Create your own classifier
- Compare classifiers with accuracy and plots
- Real time classification
* User interface for easier navigation


## Installation

First, clone the repository in a desired folder

```bash
git clone https://github.com/WexTheX/Bachelor-E2511.git
```

Secondly, install the required libraries

```bash
pip install requirements.txt
```

# Usage 


## Training and data analysis
To use the program, run the `main.py` file with the desired paramaters. The desired paramaters are in a seperate python file; `config.py`:

At the bottom of the `main.py` file:
```python
if __name__ == "__main__":

    main(**main_config)
```
In the `config.py` file:
```python
main_config = {

    # --- GLOBAL VARIABLES / FLAGS ---
    'want_feature_extraction':  0,
    'use_granular_labels':      1, 
    'want_new_CLFs':            0,
    'want_plots':               1,
    'want_pickle':              0, # Pickle the classifier, scaler and PCA objects.
    'want_offline_test':        0,
    'want_calc_exposure':       0,
    'model_selection':          ['svm', 'lr', 'ada', 'gnb', 'knn', 'rf'],
    'method_selection':         ['rs'],

    # --- DATASET & MODELING VARIABLES ---
    'variance_explained':       0.95,
    'random_seed':              42,
    'window_length_seconds':    20,
    'test_size':                0.25,
    'fs':                       800,
    'ds_fs':                    800,  # Downsampled frequency
    'cmap':                     'tab10', # Colormap for plotting
    'n_iter':                   30,   # Iterations for RandomizedSearch
    'norm_IMU':                 False,

    # --- EXPOSURE CALCULATION VARIABLES ---
     'exposures_and_limits': {'CARCINOGEN':  1000.0,
                             'RESPIRATORY': 750.0, 
                             'NEUROTOXIN':  30.0, 
                             'RADIATION':   120.0, 
                             'NOISE':       900.0, 
                             'VIBRATION':   400.0,
                             'THERMAL':     2500.0, 
                             'MSK':         500.0
                            },
    'variables': ["Timestamp","Gyr.X","Gyr.Y","Gyr.Z","Axl.X","Axl.Y","Axl.Z","Mag.X","Mag.Y","Mag.Z","Temp"],

    # --- FILE PATHS ---
    'test_file_path':           "testOnFile/testFiles",
    'prediction_csv_path':      "testOnFile",
    'clf_results_path':         "CLF results/clf_results.joblib"
}
```
 
 Global variables are boolean `True` or `False`, i.e `1` or `0`. For example if you want plots for visualization set `want_plots = 1`.

 # Settings explanation

- **`want_feature_extraction`**: Finds the features in each window (`window_length_seconds` variable changes the window length in seconds) of the dataset if `true` and stores them for easy creation of a new classifier. `main.py` has to be run with this enabled if new data is gathered.

- **`use_granular_labels`**: Separates the type of category even further (e.g, Tig steel vs Tig Aluminium instead of just Welding) if `true`
- **`want_new_CLFs`**: Creates and compares classifier models that are selected in `model_selection` and chooses the highest performing model if `true `. 
- **`want_plots`**: Shows a variety of useful plots such as the confusion matrix, PCA plot and learningcurve if `True`. Good in combination with `want_new_CLFs` for analysis.
- **`want_pickle`**: Saves the current classifier by using the `pickle` library. Good if new data is collected and `want_new_CLFs` finds a better model.
- **`want_offline_test`**: Tests the classifier and its certainty (top 3 predicted classes) for each window in all the data files in `YourDirectory/Bachelor-E2511/testOnFile/testFiles` if `true`.
- **`want_calc_exposure`**: A work in progress functionality that calculates estimated exposure to vibrations, neurotoxins etc if `true`.
- **`model_selection`**: Allows you to specify which machine learning models to use or test within the program. This enables straightforward model comparison and selection for your task.

You can choose one or multiple models from the following list:

 Code | Model Name                  
-|-
`lr`  | Logistic Regression  
`svm` | Support Vector Machine       
`knn` | K-Nearest Neighbors   
`rf`  | Random Forest     
`ada` | Adaptive Boosting (AdaBoost)
`gb`  | Gradient Boosting
`gnb` | Gaussian Naive Bayes        
       
          

To compare multiple models, simply pass a list like:

```python
model_selection = ['svm', 'lr', 'rf']
```

- **`method_selection`**:

#### Dataset & modeling variables
- **`variance_explained`**: The amount of variance from 0-1 (0-100%) to be captured by the PCA components normal is 0.95 (95%). If `variance_explained` > 1 then the program treats it as the amount of PCA components to use. I.e `variance_explained=3` means it will only use 3 principal components. This may be useful for analysis since it allows for the clusters to be plotted in 3D.
- **`random_seed`**: For validation it ensures randomness of the training-test split. 
- **`window_length_seconds`**: How long the windows should be, for this project `window_length_seconds=20` was standard.
- **`test_size`**: How much of the dataset should be used for testing, from 0-1 (0-100%). Normaly `test_size=0.25`.
- **`fs`**: The sample frequency used on the sensor.
- **`ds_fs`**: The desired downsampled frequency.
- **`cmap`**: Color setting in the `pyplot` library. Only for getting nice colors. The project used mainly `cmap=tab10`. Visit `https://matplotlib.org/stable/users/explain/colors/colormaps.html` for more info.
- **`n_iter`**: The amount of iterations to be used when using randomized search for hyperparameter optimization. The project used `n_iter=30`.
- **`norm_imu`**: Whether to use IMU x, y, z as features, `false` or the norm of x,y,z as features, `true`. Genereally for the project `norm_imu` was set to `false`.

#### Exposure
- **`exposures`**: placeholder values that connect activity to vibration
- **`variables`**: The sensor configuration used when logging
- **`test_file_path`**: Where the result from test file classification appear
- **`prediction_csv_path`**: Where results from feature extraction will be saved
- **`clf_results_path`**: Where a newly trained classifier will be saved




## Real time classification
Real time classification is done by running `realtime.py`. At the top of the file there are some settings that can be changed:

```python
device_list = ["Muse_E2511_GREY", "Muse_E2511_RED", "muse_v3_3", "muse_v3"] # List of bluetooth devices 
device_name = device_list[0]                        # Choose device to connect to from listq

window_length_sec = 20                  # Length of one window for prediction
fs = 200                                # Frequency of sensor sampling
window_size = window_length_sec * fs
real_time_window_sec = 30  
```


## Visualization
To visualize options and paramters run the `User_interface.py` file like this:

```bash
streamlit run User_interface.py
```

This should open a web browser page with the user interface running on `localhost` port `8051`.
From here one can interact with the code, providing the same functionalities as altering the code and runnning `main.py` aswell as `realtime.py`, though it's recomended to also pay attention to the terminal since only results are displayed not the progress. Just to avoid impatience.

