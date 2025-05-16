import streamlit as st
from main import main
import os
import asyncio
from realtime import RT_main, get_predictions
import time
import threading
from streamlit_autorefresh import st_autorefresh
from realtime import shutdown_event
import pandas as pd
import numpy as np
# args for main()
ML_models = ["SVM", "RF", "KNN", "GNB", "LR", "GB", "ADA"]
Search_methods = ["BS", "RS", "GS", "HGS"]
test_file_path = "testOnFile/testFiles"
prediction_csv_path = "testOnFile"
clf_results_path = "CLF results/clf_results.joblib"
cmap = "tab10"
random_seed = 420
frequencies = [25, 50, 100, 200, 400, 800]
percentages = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 2, 3]
n_iter = 30
exposures_and_limits = {'CARCINOGEN': 1000.0,
                        'RESPIRATORY': 750.0,
                        'NEUROTOXIN': 30.0,
                        'RADIATION': 120.0,
                        'NOISE': 900.0,
                        'VIBRATION': 400.0,
                        'THERMAL': 2500.0,
                        'MSK': 500.0}
variables = ["Timestamp","Gyr.X","Gyr.Y","Gyr.Z","Axl.X","Axl.Y","Axl.Z","Mag.X","Mag.Y","Mag.Z","Temp"] #Unused
norm_IMU = False

plots = {}
prediction_list = {}
result = {}


st.title("User interface")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Real time streaming", "ML model", "Results", "Test classifier on a file" , "New files/data" ])
        

## TAB NUMBER ONE ##
with tab1:
    column1,column2 = st.columns(2)
    with column1:
        if st.button("Start classifying in real time"):
            asyncio.run(RT_main())
        
        if st.button("Stop classifying in real time"):
            shutdown_event.set()

    
    with column2:
        if st.checkbox("Show classification"):
            prediction_list = get_predictions()
            #st.write(prediction_list)
            st.write({'Thu, 10 Apr 2025 13:26:14 +0000': np.str_('IDLE'),	 	
                     'Thu, 10 Apr 2025 13:26:34 +0000': np.str_('IMPA'),	 	
                     'Thu, 10 Apr 2025 13:26:54 +0000': np.str_('IMPA')})
            st_autorefresh(interval= 10 * 1000, key="test")



## TAB NUMBER TWO ##
with tab2:
    st.info("This tab is for running main function with chosen parameters")
    col1, col2 = st.columns(2)

    with col1:
        with st.expander("Model and Settings"):
            model_selection = st.multiselect("Select ML models: ", ML_models)
            method_selection = st.multiselect("Hyperparameter search methods:", Search_methods)
            Splitting_method = st.write("Validation technique: StratifiedKFOLD")
            want_feature_extraction = st.checkbox("Want feature extraction")
            want_new_CLFs = st.checkbox("Want new classifiers")
            save_joblib = st.checkbox("Save classifier and transformations with joblib")
            use_granular_labels = st.checkbox("Separate types of classes (TIG VS MAG etc)", True)
            want_offline_test = st.checkbox("Want offline test")
            want_calc_exposure = st.checkbox("Want calc exposure")
            want_plots = st.checkbox("Want plots and results")
        
        
    with col2:
        with st.expander("Preprocessing Parameters"):
            fs = st.write("Sampling frequency: 800 Hz")
            ds_fs = st.selectbox("Downsampled frequency:", frequencies, index=5)
            window_length_seconds = st.selectbox("Window length (seconds):", [20, 40])
            test_size = st.selectbox("Amount of test data", [0.25, 0.3])
            variance_explained = st.selectbox("PCA variance (%) to retain:", percentages, index=5)



    st.markdown("---")


    if st.button("Run main function"):

        config = {
            'want_feature_extraction': want_feature_extraction,
            'save_joblib': save_joblib,
            'use_granular_labels': use_granular_labels,
            'want_new_CLFs': want_new_CLFs,
            'want_plots': want_plots,
            'want_offline_test': want_offline_test,
            'want_calc_exposure': want_calc_exposure,
            'model_selection': model_selection,
            'method_selection': method_selection,
            'variance_explained': variance_explained,
            'random_seed': random_seed,
            'window_length_seconds': window_length_seconds,
            'test_size': test_size,
            'fs': fs,
            'ds_fs': ds_fs,
            'cmap': cmap,
            'test_file_path': test_file_path,
            'prediction_csv_path': prediction_csv_path,
            'clf_results_path': clf_results_path,
            'n_iter': n_iter,
            'exposures_and_limits': exposures_and_limits,
            'variables': variables,
            'norm_IMU': norm_IMU
        }

        plots, result, metrics_df = main(**config)
        
        st.session_state["plots"] = plots
        st.session_state["result"] = result

        #st.write(result)
    
    


## TAB NUMBER THREE ##
with tab3:
    st.write(plots)
    plots = st.session_state.get("plots", {})
    result = st.session_state.get("result", {})

    if want_plots and plots:
        st.write(f"Best classifier: {result['model_name']}  \n Hyperparameter search method: {result['optimalizer']}")
        st.write(f"Hyperparameter dictionary: {result['best_params']}")
        st.write(f"Accuracy: {round(result['test_accuracy'], 3)}    \n F-score: {round(result['test_f1_score'], 3)}")
    
        selected_plots = st.multiselect("Select plot(s)", plots.keys())


        
        for plot_name in selected_plots:
            st.subheader(plot_name)

            plot_obj = plots[plot_name]
            
            if isinstance(plot_obj, list):
                for i, fig in enumerate(plot_obj):
                    st.pyplot(fig)
            else:
                st.pyplot(plot_obj)

    else:
        st.info("Plots will be displayed here if 'Want plots' is checked and the main function is ran")

#  plots = {
#                      'Learning curve': fig_0,
#                      'Confusion matrix': fig_1,
#                      'PCA table': fig_list_0,
#                      'Feature importance': fig_list_1,
#                      'Scree plot': fig_2,
#                      'Biplot': fig_3,
#                      'Biplot 3D': fig_4,
#                      'Decision boundaries': fig_5
#                      }

with tab4:    
        if want_offline_test:
            predictions_path = 'testOnFile/predictions.csv' 
            df_predictions = pd.read_csv(predictions_path)
            st.write("Displaying the content of predictions.csv")
            st.dataframe(df_predictions)  

        else:
            st.info("If 'Want offline test' is checked in the ML model tab results of uploaded files will be displayed here")


        if want_calc_exposure:
            summary_path = 'testOnFile/summary.csv' 
            df_summary = pd.read_csv(summary_path)
            st.write(f"Displaying the content of {summary_path}")
            st.dataframe(df_summary)  

        upload_directory = 'testOnFile/testFiles/'
        uploaded_test_file = st.file_uploader("Upload the file(s) you want to check", type= 'txt')


        if uploaded_test_file is not None:
            
            test_file_path = os.path.join(upload_directory, uploaded_test_file.name)

            with open(test_file_path, "wb") as d:
                d.write(uploaded_test_file.getbuffer()) 
            d.close

            st.success(f"File uploaded and saved to {upload_directory}")
        

## TAB NUMBER FIVE ##
with tab5:
    st.info("This tab is for adding new data for training the ML model")

    path_granular = "Datafiles/DatafilesSeparated_Aker"
    path_combined = "Datafiles/DatafilesCombined_aker"

    # TODO Fix separated and combined
    if use_granular_labels == True:
        
        category_dirs = {
            "GrindBig": path_granular + "/GrindBig",
            "GrindMed": path_granular + "GrindMed",
            "GrindSmall": path_granular + "/GrindSmall",
            "Idle": path_granular + "/Idle",
            "Impa": path_granular + "/Impa",  
            #"SandSim": path + "/DatafilesSeparated/SandSim",
            "WeldAlTIG": path_granular + "/WeldAlTIG",
            "WeldStMAG": path_granular + "/WeldStMAG",
            "WeldStTIG": path_granular + "/WeldStTIG"
        }
    
    else:
        category_dirs = {
            "Grinding": path_combined + "/Grinding",
            "Idle": path_combined + "/Idle",
            "Impa": path_combined + "/Impa",
            "Welding": path_combined + "/Welding"
        }


    #Dropdown menu for category selection
    selected_category = st.selectbox("Select Activity Type", category_dirs.keys())

    #File uploader
    uploaded_file = st.file_uploader(f"Upload file for {selected_category}", type=["txt"])

    if uploaded_file is not None:
        target_dir = category_dirs[selected_category]

        file_path = os.path.join(target_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        f.close()
        
        st.success(f"File uploaded and saved to {file_path}")


    


