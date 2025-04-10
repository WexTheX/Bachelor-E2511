import streamlit as st
from main import main

want_feature_extraction = 1
pickle_files            = 1 # Pickle the classifier, scaler and PCA objects.
separate_types          = 1
want_plots              = 1
ML_models               = ["SVM", "RF", "KNN", "GNB", "COMPARE"]
ML_model                = "SVM"
Splitting_method        = ["StratifiedKFOLD", "TimeSeriesSplit"]
Splitting_method        = "TimeseriesSplit"

#DATASET VARIABLES

variance_explained      = 2
random_seed             = 333
window_length_seconds   = 20
test_size               = 0.25
fs                      = 800
ds_fs                   = 800
variables               = ["Timestamp","Gyr.X","Gyr.Y","Gyr.Z","Axl.X","Axl.Y","Axl.Z","Mag.X","Mag.Y","Mag.Z","Temp"]

st.title("User interface")

tab1, tab2 = st.tabs(["Real time streaming", "ML model" ])

with tab1:
    "yoo"


frequencies = [25, 50, 100, 200, 400, 800, 1600]
percentages = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 2]

with tab2:

    st.write("This tab is for creating a new classifier")

    with st.form(key="testing"):
        col1, col2, col3 = st.columns(3)

        with col1:
            #ML model 
            model_selection = st.multiselect("Select ML model: ",ML_models)
            
            #Additional settings: 
            want_feature_extraction = st.checkbox("Want feature extraction")

            pickle_files = st.checkbox("Want to save classifier and transformations with pickle")
            separate_types = st.checkbox("Want to separate types of classes (TIG VS MAG etc)")
            want_plots = st.checkbox("Want plots")

            Splitting_method = st.selectbox("What validation technique to use: ", ["StratifiedKFOLD", "TimeSeriesSplit"])
            method_selection = st.multiselect("What method(s) to use to find the best hyperparameters: ",['GridSearchCV', 'BayesSearchCV0', 'RandomizedSearchCV'])
            

        

        with col2:
            st.header("Settings")
            frequency = st.selectbox("Frequency used for sampling: ", frequencies)
            downsample = st.selectbox("Desired downsampled frequency: ", frequencies)
            window_length_seconds = st.selectbox("Desired window length in seconds: ", [20,40])
            variance_explained = st.selectbox("Desired variance to be captured by PCA in percentage:", percentages)
        

        with col3:
            if st.form_submit_button("Create classifier"):
            
                
                main(want_feature_extraction, pickle_files, 
                    separate_types, want_plots, Splitting_method, 
                    model_selection, method_selection, variance_explained ,
                    random_seed ,window_length_seconds ,test_size , fs, ds_fs )
                        
            



