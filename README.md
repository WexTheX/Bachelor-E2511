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
    'separate_types':           1, 
    'want_new_CLFs':            0,
    'want_plots':               1,
    'want_pickle':              0, # Pickle the classifier, scaler and PCA objects.
    'want_offline_test':        0,
    'want_calc_exposure':       0,
    'model_selection':          ['svm', 'lr', 'ada', 'gnb', 'knn', 'rf'],
    'method_selection':         ['rs'],

    # --- DATASET & MODELING VARIABLES ---
    'variance_explained':       0.95,
    'random_seed':              4201,
    'window_length_seconds':    20,
    'test_size':                0.25,
    'fs':                       800,
    'ds_fs':                    800,  # Downsampled frequency
    'cmap':                     'tab10', # Colormap for plotting
    'n_iter':                   30,   # Iterations for RandomizedSearch
    'norm_IMU':                 False,

    # --- EXPOSURE CALCULATION VARIABLES ---
    'exposures': [
        'CARCINOGEN', 'RESPIRATORY', 'NEUROTOXIN', 'RADIATION', 'NOISE', 'VIBRATION', 'THERMAL', 'MSK'
    ],
    'safe_limit_vector': [1000.0, 750.0, 30.0, 120.0, 900.0, 400.0, 2500.0, 400.0], 
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

- **`separate_types`**: Separates the type of category even further (e.g, Tig steel vs Tig Aluminium instead of just Welding) if `true`
- **`want_new_CLFs`**: Creates and compares classifier models that are selected in `model_selection` and chooses the highest performing model if `true `. 
- **`want_plots`**: Shows a variety of useful plots such as the confusion matrix, PCA plot and learningcurve if `True`. Good in combination with `want_new_CLFs` for analysis.
- **`want_pickle`**: Saves the current classifier by using the `pickle` library. Good if new data is collected and `want_new_CLFs` finds a better model.
- **`want_offline_test`**: Tests the classifier and its certainty for each window in all the data files in `YourDirectory/Bachelor-E2511/`
- **`want_calc_exposure`**: 
- **`model_selection`**:
- **`method_selection`**:

#### Dataset & modeling variables
- **`variance_explained`**:
- **`random_seed`**:
- **`window_length_seconds`**:
- **`test_size`**:
- **`fs`**:
- **`ds_fs`**:
- **`cmap`**:
- **`n_iter`**:
- **`norm_imu`**:

#### Exposure
- **`exposures`**:
- **`safe_limit_vector`**:
- **`variables`**:
- **`test_file_path`**:
- **`prediction_csv_path`**: y
- **`clf_results_path`**:

 #### Visualization
 To better visualize options and paramters that you can change run the `User_interface.py` file like this:

 ```bash
 streamlit run User_interface.py
 ```
 






