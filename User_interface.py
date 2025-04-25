import streamlit as st
from main import main
import os
import asyncio
from realtime import RT_main, get_predictions
import time
import threading
from streamlit_autorefresh import st_autorefresh
from realtime import shutdown_event

ML_models = ["SVM", "RF", "KNN", "GNB", "LR", "GB", "ADA"]
Search_methods = ["BS", "RS", "GS", "HGS"]
test_file_path = "testOnFile/testFiles"
prediction_csv_path = "testOnFile"
clf_results_path = "CLF results/clf_results.joblib"
cmap = "tab10"
random_seed = 420
plots = []
prediction_list = {}
frequencies = [25, 50, 100, 200, 400, 800, 1600]
percentages = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 2, 3]
n_iter = 30
exposures = [
    'CARCINOGEN', 'RESPIRATORY', 'NEUROTOXIN', 'RADIATION', 'NOISE', 'VIBRATION', 'THERMAL', 'MSK'
]
safe_limit_vector = [1000.0, 750.0, 30.0, 120.0, 900.0, 400.0, 2500.0, 400]
variables = ["Timestamp","Gyr.X","Gyr.Y","Gyr.Z","Axl.X","Axl.Y","Axl.Z","Mag.X","Mag.Y","Mag.Z","Temp"] #Unused


st.title("User interface")

tab1, tab2, tab3, tab4 = st.tabs(["Real time streaming", "ML model", "Results", "New files/data" ])
        

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
            st.write(prediction_list)       
            st_autorefresh(interval= 10 * 1000, key="test")



## TAB NUMBER TWO ##
with tab2:
    st.info("This tab is for running main function with chosen parameters")
    col1, col2 = st.columns(2)

    with col1:
        with st.expander("Model and Settings"):
            model_selection = st.multiselect("Select ML models: ", ML_models)
            method_selection = st.multiselect("Hyperparameter search methods:", Search_methods)
            Splitting_method = st.selectbox("Validation technique:", ["StratifiedKFOLD", "TimeSeriesSplit"])
            want_feature_extraction = st.checkbox("Want feature extraction")
            want_new_CLFs = st.checkbox("Want new classifiers")
            want_pickle = st.checkbox("Save classifier and transformations with pickle")
            separate_types = st.checkbox("Separate types of classes (TIG VS MAG etc)", True)
            want_offline_test = st.checkbox("Want offline test")
            want_calc_exposure = st.checkbox("Want calc exposure")
            want_plots = st.checkbox("Want plots and results")
        
        
    with col2:
        with st.expander("Preprocessing Parameters"):
            fs = st.selectbox("Sampling frequency:", frequencies, index=5)
            ds_fs = st.selectbox("Downsampled frequency:", frequencies, index=3)
            window_length_seconds = st.selectbox("Window length (seconds):", [20, 40])
            test_size = st.selectbox("Amount of test data (%)", [0.25, 0.3])
            variance_explained = st.selectbox("PCA variance (%) to retain:", percentages)



    st.markdown("---")


    if st.button("Run main function"):
        plots, result, accuracy_list = main(
            want_feature_extraction,want_pickle, 
            separate_types, want_new_CLFs, want_plots, want_offline_test, want_calc_exposure,
            model_selection, method_selection, variance_explained,
            random_seed ,window_length_seconds, test_size, fs, ds_fs, 
            cmap, test_file_path, prediction_csv_path, clf_results_path,
            n_iter, exposures, safe_limit_vector, variables
        )
        #st.write(result)
    
    


## TAB NUMBER THREE ##
with tab3:
    if want_plots and plots != []:
        st.write(f"Best clf and hyperparam search method: {result['model_name']}, {result['optimalizer']}")
        st.write(f"Accuracy: {max(accuracy_list)}")

        st.pyplot(plots[0][0])
        st.pyplot(plots[0][1])
        st.pyplot(plots[2])

        if variance_explained == 2:
            st.pyplot(plots[3])
      
    else:
        st.info("Plots will be displayed here after training if 'Want plots' is checked.")



## TAB NUMBER FOUR ##
with tab4:
    st.info("This tab is for adding new data for training the ML model")
    category_dirs = {
        "GrindBig": "Preprocessing/DatafilesSeparated/GrindBig",
        "GrindMed": "Preprocessing/DatafilesSeparated/GrindMed",
        "GrindSmall": "Preprocessing/DatafilesSeparated/GrindSmall",
        "Idle": "Preprocessing/DatafilesSeparated/Idle",
        "Impa": "Preprocessing/DatafilesSeparated/Impa",  
        "SandSim": "Preprocessing/DatafilesSeparated/SandSim",
        "WeldAlTIG": "Preprocessing/DatafilesSeparated/WeldAlTIG",
        "WeldStMAG": "Preprocessing/DatafilesSeparated/WeldStMAG",
        "WeldStTIG": "Preprocessing/DatafilesSeparated/WeldStTIG"
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


    


