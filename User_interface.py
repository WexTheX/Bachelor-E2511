import streamlit as st
from main import main
import os

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

tab1, tab2, tab3 = st.tabs(["Real time streaming", "ML model", "New files/data" ])

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
                    random_seed ,window_length_seconds ,test_size , fs, ds_fs)
                        










## TAB NUMBER THREEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE ##

with tab3:
    c1,c2,c3 = st.columns(3)
    

    with c1:
        uploaded_file = st.file_uploader("GrindBig", type=["txt"])

        if uploaded_file is not None:
            target_dir = "Preprocessing/DatafilesSeparated/GrindBig"
            
            file_path = os.path.join(target_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())       
            
                    
                    
            st.success(f"File uploaded and saved to {file_path}")
    



    with c2:
        uploaded_file2 = st.file_uploader("GrindMed", type=["txt"])

        if uploaded_file2 is not None:
            target_dir2 = "Preprocessing/DatafilesSeparated/GrindMed"
            
            file_path2 = os.path.join(target_dir2, uploaded_file2.name)
            with open(file_path2, "wb") as f:
                f.write(uploaded_file2.getbuffer())       
            
                    
                    
            st.success(f"File uploaded and saved to {file_path2}")
    



    with c3:
        uploaded_file3 = st.file_uploader("GrindSmal", type=["txt"])

        if uploaded_file3 is not None:
            target_dir3 = "Preprocessing/DatafilesSeparated/GrindSmall"
            
            file_path3 = os.path.join(target_dir3, uploaded_file3.name)
            with open(file_path3, "wb") as f:
                f.write(uploaded_file3.getbuffer())       
            
                    
                    
            st.success(f"File uploaded and saved to {file_path3}")


    r2c1, r2c2, r2c3 = st.columns(3)

    with r2c1:
        uploaded_file4 = st.file_uploader("Idle", type=["txt"])

        if uploaded_file4 is not None:
            target_dir4 = "Preprocessing/DatafilesSeparated/Idle"
            
            file_path4 = os.path.join(target_dir4, uploaded_file4.name)
            with open(file_path4, "wb") as f:
                f.write(uploaded_file4.getbuffer())       
            
                    
                    
            st.success(f"File uploaded and saved to {file_path4}")


    

    with r2c2:
        uploaded_file5 = st.file_uploader("Impa", type=["txt"])

        if uploaded_file5 is not None:
            target_dir5 = "Preprocessing/DatafilesSeparated/Idle"
            
            file_path5 = os.path.join(target_dir5, uploaded_file5.name)
            with open(file_path5, "wb") as f:
                f.write(uploaded_file5.getbuffer())       
            
                    
                    
            st.success(f"File uploaded and saved to {file_path5}")
    
    with r2c3:
        uploaded_file6 = st.file_uploader("SandSim", type=["txt"])

        if uploaded_file6 is not None:
            target_dir6 = "Preprocessing/DatafilesSeparated/SandSim"
            
            file_path6 = os.path.join(target_dir6, uploaded_file6.name)
            with open(file_path6, "wb") as f:
                f.write(uploaded_file6.getbuffer())       
            
                    
                    
            st.success(f"File uploaded and saved to {file_path6}")



    

    r3c1, r3c2,r3c3 = st.columns(3)


    with r3c1:
        uploaded_file7 = st.file_uploader("WeldAlTIG", type=["txt"])

        if uploaded_file7 is not None:
            target_dir7 = "Preprocessing/DatafilesSeparated/WeldAlTIG"
            
            file_path7 = os.path.join(target_dir7, uploaded_file7.name)
            with open(file_path7, "wb") as f:
                f.write(uploaded_file7.getbuffer())       
            
                    
                    
            st.success(f"File uploaded and saved to {file_path7}")


    
    with r3c2:
        uploaded_file8 = st.file_uploader("WeldStMAG", type=["txt"])

        if uploaded_file8 is not None:
            target_dir8 = "Preprocessing/DatafilesSeparated/WeldStMAG"
            
            file_path8 = os.path.join(target_dir8, uploaded_file8.name)
            with open(file_path8, "wb") as f:
                f.write(uploaded_file8.getbuffer())       
            
                    
                    
            st.success(f"File uploaded and saved to {file_path8}")
    





    with r3c3:
        uploaded_file9 = st.file_uploader("WeldStTIG", type=["txt"])

        if uploaded_file9 is not None:
            target_dir9 = "Preprocessing/DatafilesSeparated/WeldStTIG"
            
            file_path9 = os.path.join(target_dir9, uploaded_file9.name)
            with open(file_path9, "wb") as f:
                f.write(uploaded_file9.getbuffer())       
            
                    
                    
            st.success(f"File uploaded and saved to {file_path9}")




    


