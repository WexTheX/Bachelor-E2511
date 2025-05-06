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
To use the program, run the `main.py` file with the desired paramaters. The desired paramaters are at the bottom of the script:

```python

if __name__ == "__main__":

    ''' GLOBAL VARIABLES '''


    want_feature_extraction = 0
    want_pickle             = 0 
    separate_types          = 1 
    want_plots              = 1
    want_offline_test       = 1
    want_calc_exposure      = 1

    model_selection         = ['svm']
    method_selection        = ['rs']

    ''' DATASET VARIABLES '''

    variance_explained      = 3
    random_seed             = 420
    window_length_seconds   = 20
    test_size               = 0.25
    fs                      = 800
    ds_fs                   = 200
    cmap                    = 'tab10'

    ''' LOAD PATH NAMES'''
    test_file_path = "testOnFile/testFiles"
    prediction_csv_path = "testOnFile"

```
 
 Global variables are boolean `True` or `False`, i.e `1` or `0`. For example if you want plots for visualization set `want_plots = 1`.

 #### Visualization
 To better visualize options and paramters that you can change run the `User_interface.py` file like this:
 ```bash
 streamlit run User_interface.py
 ```
 






