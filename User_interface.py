import streamlit as st
from main import main
import os
import asyncio
from realtime import RT_main, get_predictions
import time
import threading


ML_models = ["SVM", "RF", "KNN", "GNB", "COMPARE"]
Search_methods = ['GridSearchCV', 'BayesSearchCV0', 'RandomizedSearchCV', 'bs', 'rs']
test_file_path = "testOnFile/testFiles"
prediction_csv_path = "testOnFile"
random_seed = 420
plots = []
prediction_list = {}


st.title("User interface")

tab1, tab2, tab3, tab4 = st.tabs(["Real time streaming", "ML model", "Results", "New files/data" ])

if "predictions" not in st.session_state:
    st.session_state.predictions = {}


def start_ble_loop():
    asyncio.run(RT_main())



def display_predictions():
    prediction_placeholder = st.empty()
    while True:
        predictions = get_predictions()
        if predictions != st.session_state.predictions:
            st.session_state.predictions = predictions
            prediction_text = "\n".join([f"{k}: {v}" for k, v in predictions.items()])
            prediction_placeholder.text(prediction_text)
        time.sleep(1)


with tab1:
    if st.button("Start classifying in real time"):
        ble_thread = threading.Thread(target=start_ble_loop, daemon=True)
        ble_thread.start()

        display_predictions()
        

        
        

frequencies = [25, 50, 100, 200, 400, 800, 1600]
percentages = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 2]


with tab2:
    st.info("This tab is for running main function with chosen parameters")
    col1, col2 = st.columns(2)

    with col1:
        with st.expander("Model and Settings"):
            model_selection = st.multiselect("Select ML models: ", ML_models)
            method_selection = st.multiselect("Hyperparameter search methods:", Search_methods)
            Splitting_method = st.selectbox("Validation technique:", ["StratifiedKFOLD", "TimeSeriesSplit"])
            want_feature_extraction = st.checkbox("Want feature extraction")
            want_pickle = st.checkbox("Save classifier and transformations with pickle")
            separate_types = st.checkbox("Separate types of classes (TIG VS MAG etc)")
            want_offline_test = st.checkbox("Want offline test")
            want_calc_exposure = st.checkbox("Want calc exposure")
            want_plots = st.checkbox("Want plots and results")
        
        
    with col2:
        with st.expander("Preprocessing Parameters"):
            fs = st.selectbox("Sampling frequency:", frequencies)
            ds_fs = st.selectbox("Downsampled frequency:", frequencies)
            window_length_seconds = st.selectbox("Window length (seconds):", [20, 40])
            test_size = st.selectbox("Amount of test data (%)", [0.25, 0.3])
            variance_explained = st.selectbox("PCA variance (%) to retain:", percentages)



    st.markdown("---")


    if st.button("Run main function"):
        plots, result, accuracy_list = main(
            want_feature_extraction, want_pickle, 
            separate_types, want_plots, want_offline_test, want_calc_exposure,
            model_selection, method_selection, variance_explained,
            random_seed ,window_length_seconds, test_size, fs, ds_fs, 
            "tab10", test_file_path, prediction_csv_path
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


    


