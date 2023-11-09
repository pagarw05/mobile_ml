# App to predict the class of mobile price range
# Using a pre-trained ML model in Streamlit

# Import libraries
import streamlit as st
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

st.title('Range of Mobile Prices: A Machine Learning App') 

# Display the image
st.image('mobile_image.jpg', width = 650)

st.write("This app uses multiple inputs to predict the class of mobile price range. " 
         "Use the following form or upload your dataset to get started!") 

# Reading the pickle files that we created before 
# Decision Tree
dt_pickle = open('dt_mobile.pickle', 'rb') 
dt_model = pickle.load(dt_pickle) 
dt_pickle.close()

# Random Forest
rf_pickle = open('rf_mobile.pickle', 'rb') 
rf_model = pickle.load(rf_pickle) 
rf_pickle.close()

# Loading default dataset
default_df = pd.read_csv('mobile.csv')

with st.form('user_inputs'): 
  battery_power = st.number_input('Battery power', min_value = 0, value = 1500, step = 10)
  blue = st.radio('Has bluetooth or not?', options = ['Yes', 'No']) 
  clock_speed = st.number_input('Speed at which microprocessor executes instructions', min_value = 0.0, value = 0.5)
  dual_sim = st.radio('Has dual sim support or not?', options = ['Yes', 'No']) 
  fc = st.number_input('Front camera mega pixels', min_value = 0, value = 2)
  four_g = st.radio('Has 4G or not?', options = ['Yes', 'No'])
  int_memory = st.number_input('Internal memory in GB', min_value = 0, value = 25)
  m_dep = st.number_input('Mobile depth in cm', min_value = 0.0, value = 0.5)
  mobile_wt = st.number_input('Weight of mobile phone', min_value = 0, value = 100, step = 10)
  n_cores = st.number_input('Number of cores of processor', min_value = 0, value = 5)
  pc = st.number_input('Primary camera mega pixels', min_value = 0, value = 5)
  px_height = st.number_input('Pixel resolution height', min_value = 0, value = 500, step = 10)
  px_width = st.number_input('Pixel resolution width', min_value = 0, value = 1000, step = 10)
  ram = st.number_input('Random Access Memory (RAM) in MB', min_value = 0, value = 3500, step = 10)
  sc_h = st.number_input('Screen height of mobile in cm', min_value = 0, value = 10)
  sc_w = st.number_input('Screen width of mobile in cm', min_value = 0, value = 6)
  talk_time = st.number_input('Longest time that a single battery charge will last', min_value = 0, value = 10)
  three_g = st.radio('Has 3G or not?', options = ['Yes', 'No'])
  touch_screen = st.radio('Has touch screen or not?', options = ['Yes', 'No'])
  wifi = st.radio('Has wifi or not?', options = ['Yes', 'No'])
  ml_model = st.selectbox('Select Machine Learning Model for Prediction', options = ['Decision Tree', 'Random Forest'],
                          placeholder = 'Choose an option') 
  st.form_submit_button() 

encode_df = default_df.copy()
encode_df = encode_df.drop(columns = ['price_range'])
# Combine the list of user data as a row to default_df
encode_df.loc[len(encode_df)] = [battery_power, blue, clock_speed, dual_sim, fc, four_g, int_memory, m_dep, mobile_wt, n_cores,
                                 pc, px_height, px_width, ram, sc_h, sc_w, talk_time, three_g, touch_screen, wifi]
# Create dummies for encode_df
cat_var = ['blue', 'dual_sim', 'four_g', 'three_g', 'touch_screen', 'wifi']
encode_dummy_df = pd.get_dummies(encode_df, columns = cat_var)
# Extract encoded user data
user_encoded_df = encode_dummy_df.tail(1)

st.subheader("Predicting Class of Mobile Price Range")

if ml_model == 'Decision Tree':
    # Using DT to predict() with encoded user data
    new_prediction_dt = dt_model.predict(user_encoded_df)
    new_prediction_prob_dt = dt_model.predict_proba(user_encoded_df).max()
    # Show the predicted cost range on the app
    st.write("Decision Tree Prediction: {}".format(*new_prediction_dt))
    st.write("Prediction Probability: {:.0%}".format(new_prediction_prob_dt))

    # Showing additional items
    st.subheader("Prediction Performance")
    tab1, tab2, tab3 = st.tabs(["Feature Importance", "Confusion Matrix", "Classification Report"])
    with tab1:
        st.image('dt_feature_imp.svg')
    with tab2:
        st.image('dt_confusion_mat.svg')
    with tab3:
        df = pd.read_csv('dt_class_report.csv', index_col = 0)
        st.dataframe(df)

else:
    # Using RF to predict() with encoded user data
    new_prediction_rf = rf_model.predict(user_encoded_df)
    new_prediction_prob_rf = rf_model.predict_proba(user_encoded_df).max()
    # Show the predicted cost range on the app
    st.write("Random Forest Prediction: {}".format(*new_prediction_rf))
    st.write("Prediction Probability: {:.0%}".format(new_prediction_prob_rf))

    # Showing additional items
    st.subheader("Prediction Performance")
    tab1, tab2, tab3 = st.tabs(["Feature Importance", "Confusion Matrix", "Classification Report"])
    with tab1:
        st.image('rf_feature_imp.svg')
    with tab2:
        st.image('rf_confusion_mat.svg')
    with tab3:
        df = pd.read_csv('rf_class_report.csv', index_col = 0)
        st.dataframe(df)
