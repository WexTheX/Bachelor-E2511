import streamlit as st


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

tab1, tab2 = st.tabs(["ML model", "Sensor and sampling settings"])

with tab1:
    #ML model 
    ML_model = st.multiselect("Select ML model: ",ML_models)
    
    #Additional settings: 

    pickle_files = st.checkbox("Want to save classifier and transformations with pickle")
    want_plots = st.checkbox("Want plots")
    want_feature_extraction = st.checkbox("Want feature extraction")

frequencies = [25, 50, 100, 200, 400, 800, 1600]
percentages = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]

with tab2:
    with st.form(key="Sensor and sampling settings"):
        frequency = st.selectbox("Frequency used for sampling: ", frequencies)
        downsample = st.selectbox("Desired downsampled frequency: ", frequencies)
        window_length_seconds = st.number_input("Desired window length in seconds: ")
        variance_explained = st.selectbox("Desired variance to be captured by PCA in percentage:", percentages)
        st.form_submit_button()
