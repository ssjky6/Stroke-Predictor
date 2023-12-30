#Author: Snehpriya Jha

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Stroke Prediction App
This Web App is used to predict whether you are likely to get a stroke based on the input parameters like gender, age, various diseases, and smoking status. You can find the entire source code for the models and web deployment [here](https://github.com/ssjky6/Stroke-Predictor)

""")

st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Example CSV input file](https://github.com/ssjky6/Stroke-Predictor/main/Data/example.csv)
""")

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        gender = st.sidebar.selectbox('Gender',('Male','Female','Other'))
        age = st.sidebar.slider('Age(years)')
        hypertension = st.sidebar.slider('Hypertension',max_value=1)
        heart_disease = st.sidebar.slider('Heart Disease',max_value=1)
        ever_married = st.sidebar.selectbox('Married',('Yes','No'))
        work_type = st.sidebar.selectbox('Work Type',('Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'))
        Residence_type = st.sidebar.selectbox('Residence',('Urban', 'Rural'))
        avg_glucose_level = st.sidebar.slider('Average Glucose Level', max_value=300)
        bmi = st.sidebar.slider('Body mass Index')
        smoking_status = st.sidebar.selectbox('Smoking Status',('formerly smoked', 'never smoked', 'smokes', 'Unknown'))
        data = {'id': 5555,
                'gender': gender,
                'age': age,
                'hypertension': hypertension,
                'heart_disease': heart_disease,
                'ever_married': ever_married,
                'work_type': work_type,
                'Residence_type': Residence_type,
                'avg_glucose_level': avg_glucose_level,
                'bmi': bmi,
                'smoking_status': smoking_status}
        features = pd.DataFrame(data, index=[0])
        return features
    df = user_input_features()

# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase

# Encoding of ordinal features
#In this analysis, since we are taking the data as a whole, candidate_id is not required.

df.drop('id', axis = 1, inplace = True)

dict1 = {'Male' : 0, 'Female' : 1, 'Other' : 2}
df['gender'] = [dict1[item] for item in df['gender']]

dict2 = {'Yes' : 1, 'No' : 0}
df['ever_married'] = [dict2[item] for item in df['ever_married']]

dict3 = {'Private' : 1, 'Self-employed': 2, 'Govt_job': 3, 'children':4, 'Never_worked':0}
df['work_type'] = [dict3[item] for item in df['work_type']]

dict4 = {'Urban' : 1, 'Rural' : 0}
df['Residence_type'] = [dict4[item] for item in df['Residence_type']]

dict5 = {'formerly smoked':1, 'never smoked':2, 'smokes':3, 'Unknown':0}
df['smoking_status'] = [dict5[item] for item in df['smoking_status']]



#For data features, with normal/gaussian seeming distribution, we will fill the missing values with random numbers in the range between (mean - 25%, mean + 25%) (approximately).
#For data features, with modal frequency greater than 50%, we will fill the missing values with modal value.

import random
df.bmi.fillna(np.random.uniform(25,29),inplace=True)

df = df[:1] # Selects only the first row (the user input data)

# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df)

# Reads in saved classification model
load_clf = pickle.load(open('strokes_clf.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)

if st.button("Predict"):
    global result
    result = prediction_proba
    p1 = result[0][0]*100
    p2 = result[0][1]*100
    st.success("The person has "+str(p1)+"% chance of getting a stroke and "+str(p2)+"% chance of not getting a stroke.")
