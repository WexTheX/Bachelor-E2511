from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
from typing import List, Dict, Any, Tuple, Sequence, Optional


main_config = {

    # --- GLOBAL VARIABLES / FLAGS ---
    'want_feature_extraction':  1,
    'separate_types':           1,
    'want_new_CLFs':            1,
    'want_plots':               1,
    'save_joblib':              0, # Pickle the classifier, scaler and PCA objects.
    'want_offline_test':        0,
    'want_calc_exposure':       0,
    'model_selection':          ['lr', 'svm', 'knn'
                                 , 'rf', 'gb', 'ada'
                                 , 'gnb'
                                ],
    'method_selection':         ['rs'],

    # --- DATASET & MODELING VARIABLES ---
    'variance_explained':       0.95,
    'random_seed':              42,
    'window_length_seconds':    20,
    'test_size':                0.25,
    'fs':                       800,
    'ds_fs':                    800,  # Downsampled frequency, DS is WIP bc filtering
    'cmap':                     'tab10', # Colormap for plotting
    'n_iter':                   30,   # Iterations for RandomizedSearch
    'norm_IMU':                 False,

    # --- EXPOSURE CALCULATION VARIABLES ---
    'exposures': [
        'CARCINOGEN', 'RESPIRATORY', 'NEUROTOXIN', 'RADIATION', 'NOISE', 'VIBRATION', 'THERMAL', 'MSK',
    ],
    'safe_limit_vector': [1000.0, 750.0, 30.0, 120.0, 900.0, 400.0, 2500.0, 400.0, 99.0], 
    'variables': ["Timestamp","Gyr.X","Gyr.Y","Gyr.Z","Axl.X","Axl.Y","Axl.Z","Mag.X","Mag.Y","Mag.Z","Temp"],

    # --- FILE PATHS ---
    'test_file_path':           "testOnFile/testFiles",
    'prediction_csv_path':      "testOnFile",
    'clf_results_path':         "CLF results/clf_results.joblib"
}

def setupML():

    random_seed = 420
    num_folds = 3

    base_params =  {'class_weight': 'balanced', 
                    'random_state': random_seed}

    SVM_base    = svm.SVC(**base_params, probability=True)
    RF_base     = RandomForestClassifier(**base_params)
    KNN_base    = KNeighborsClassifier()
    GNB_base    = GaussianNB()
    LR_base     = LogisticRegression(**base_params)
    GB_base     = GradientBoostingClassifier(random_state=random_seed)
    ADA_base    = AdaBoostClassifier(estimator=DecisionTreeClassifier(), random_state=random_seed)

    SVM_param_grid = {
        "C":                    [0.01, 0.1,
                                 1.0, 10.0, 100.0
                                ],
        "kernel":               ["linear", "poly", "rbf", "sigmoid"],
        "gamma":                [0.01, 0.1, 1, 10.0, 100.0],
        "coef0":                [0.0, 0.5, 1.0],
        "degree":               [2, 3, 4, 5]
    }

    RF_param_grid = {
        'n_estimators':         [50, 100, 200],         # Number of trees in the forest
        'max_depth':            [10, 20, 30, None],     # Maximum depth of each tree
        'min_samples_split':    [2, 5, 10],             # Minimum samples required to split a node
        'min_samples_leaf':     [1, 2, 4],              # Minimum samples required in a leaf node
        'max_features':         ['sqrt', 'log2'],       # Number of features considered for splitting
        # 'bootstrap':            [True, False],          # Whether to use bootstrapped samples
        'criterion':            ['gini', 'entropy']     # Splitting criteria
    }

    KNN_param_grid = {
        'algorithm':            ['ball_tree', 'kd_tree', 'brute'], 
        # 'leaf_size':          [30], 
        # 'metric':             ['minkowski'], 
        # 'metric_params':      [None], 
        # 'n_jobs':             [None], 
        'n_neighbors':          [3, 4, 5], 
        'p':                    [1, 2], 
        'weights':              ['uniform', 'distance']
    }

    GNB_param_grid = {
        'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
    }

    LR_param_grid = {
        'C':                    [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],             #
        # 'dual':                 [False],                                    # Dual or Primal formulation
        # 'fit_intercept':        [True],                                     # Constant added to function (bias)             
        # 'intercept_scaling':    [1],                                        # Only useful when Solver = liblinear, fit_intercept = true
        # 'l1_ratio':             [None],                                     # ???
        'max_iter':             [100],                                      # Max iterations for solver to converge
        # 'multi_class':          ['deprecated'],                             # Deprecated
        # 'n_jobs':               [None],                                     # Amount of jobs that can run at the same time, (also set in CV, error if both)
        # 'penalty':              ['l1', 'l2', 'elasticnet', None],           # Norm of the penalty 
        # 'solver':               ['lbfgs', 'newton-cg', 'sag', 'saga'],      # Algorithm for optimization problem
        'tol':                  [0.0001],                                   # Tolerance for stopping criteria
        #'warm_start':           [False]                                      # Reuse previous calls solution
    }

    GB_param_grid = {
        'n_estimators':         [100, 200, 300],
        'learning_rate':        [0.01, 0.05, 0.1],
        'max_depth':            [3, 5, 7],
        'min_samples_split':    [2, 5, 10],
        'min_samples_leaf':     [1, 2, 4],
        'subsample':            [0.6, 0.8, 1.0],
        'max_features':         ['sqrt', 'log2', None],
    }
    
    ADA_param_grid = {
        'n_estimators':                 [50, 100, 200],
        'learning_rate':                [0.01, 0.1, 1.0],
        'estimator__max_depth':         [1, 3, 5],
        'estimator__min_samples_split': [2, 5]
    }

    model_names = {
        'SVM':  'Support Vector Machine', 
        'RF':   'Random Forest',
        'KNN':  'K-Nearest Neighbors',
        'GNB':  'Gaussian Naive Bayes',
        'LR':   'Logistic Regression',
        'GB':   'Gradient Boosting',
        'ADA':  'AdaBoost'
    }

    models = {
        'SVM':  (SVM_base,  SVM_param_grid), 
        'RF':   (RF_base,   RF_param_grid),
        'KNN':  (KNN_base,  KNN_param_grid),
        'GNB':  (GNB_base,  GNB_param_grid),
        'LR':   (LR_base,   LR_param_grid),
        'GB':   (GB_base,   GB_param_grid),
        'ADA':  (ADA_base,  ADA_param_grid)
    }

    optimization_methods = {
        'BS':   'BayesSearchCV',
        'RS':   'RandomizedSearchCV',
        'GS':   'GridSearchCV',
        'HGS':  'HalvingGridSearchCV'
    }
    
    search_kwargs = {
        'n_jobs':              -1, 
        'verbose':             0,
        'cv':                  StratifiedKFold(n_splits=num_folds),
        'scoring':             'f1_weighted',
        'return_train_score':  True
    }

    return model_names, models, optimization_methods, search_kwargs

def loadDataset(separate_types: bool,
                norm_IMU:       bool,
                cmap:           Colormap
                ) -> Tuple[str, str, List[str], List[str]]:
    
    if separate_types == True:
        
        path            = "Datafiles/DatafilesSeparated" 
        output_path     = "OutputFiles/Separated/"

        labels = [
            'GRINDBIG', 'GRINDSMALL',
            'IDLE', 'IMPA', 'GRINDMED',
            'WELDALTIG', 'WELDSTMAG', 'WELDSTTIG'
        ]

    if separate_types == False:

        path            = "Datafiles/DatafilesCombined"
        output_path     = "OutputFiles/Combined/"

        labels = [
            'IDLE', 'GRINDING', 'IMPA', 'WELDING'
        ]

    if norm_IMU == True:
    
        time_feature_suffixes   = ['mean', 'sd', 'mad', 'max', 'min', 'energy', 'entr', 'iqr', 'kurt', 'skew', 'corr']
        sensors                 = ['accel_norm', 'gyro_norm', 'mag_norm', 'temp']

        freq_feature_suffixes   = ['psdmean', 'psdmax', 'psdmin', 'psdmax(Hz)']  
        freq_sensors            = ['accel_norm', 'gyro_norm', 'mag_norm']
    
    if norm_IMU == False:

        time_feature_suffixes   = ['mean', 'sd', 'mad', 'max', 'min', 'energy', 'entr', 'iqr', 'kurt', 'skew', 'corr']
        sensors                 = ['accel_X', 'accel_Y', 'accel_Z', 'gyro_X', 'gyro_Y', 'gyro_Z', 'mag_X', 'mag_Y', 'mag_Z', 'temp']

        freq_feature_suffixes   = ['psdmean', 'psdmax', 'psdmin', 'psdmax(Hz)']  
        freq_sensors            = ['accel_X', 'accel_Y', 'accel_Z', 'gyro_X', 'gyro_Y', 'gyro_Z', 'mag_X', 'mag_Y', 'mag_Z']

    original_feature_names: List[str] = []

    # Add time features
    for sensor in sensors:
        for suffix in time_feature_suffixes:
            original_feature_names.append(f"{suffix}_{sensor}")

    # Add frequency features
    for sensor in freq_sensors:
        for suffix in freq_feature_suffixes:
            original_feature_names.append(f"{suffix}_{sensor}") 

    num_labels      = len(labels)
    cmap_name       = plt.get_cmap(cmap, num_labels)
    label_mapping   = {label: cmap_name(i) for i, label in enumerate(labels)}

    return path, output_path, labels, original_feature_names, cmap_name, label_mapping
