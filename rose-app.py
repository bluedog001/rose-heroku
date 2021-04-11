import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Enrollment Prediction App
This app predicts the **Enrollment probabilly** according to addmition information
""")

st.sidebar.header('User Input Features')

#st.sidebar.markdown("""
#[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
#""")

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    nbofRow = len(input_df)
else:
    def user_input_features():
        Gender = st.sidebar.selectbox('Gender',('M','F','U'))
        Ethnicity = st.sidebar.selectbox('Ethnicity',('Hispanic','African American','Not Available','White','Asian','American Indian','Hawaiian/Pac Islnd'))
        AcadPlan = st.sidebar.selectbox('Acad.Plan',('DDUALCRED','DPSYC-BA','DACCT-BBA','DCSIT-BA','DGNBU-BBA','DPBHLTH-BS','DCRJS-BS', 'DFINA-BBA', 'DBIOL-BS', 'DCDFS-BS',
        'DBIOL-BA', 'DMATH-BA', 'DUNDECIDED', 'DBUND', 'DPBHLTH-BA',
        'DCOMM-BA', 'DHPMGT-BBA', 'DHROB-BBA', 'DPOLS-BA', 'DHSML-BS',
        'DLSCM-BS', 'DSOCI-BA', 'DINDE-BS', 'DSOCI-MNU', 'DSPAN-MNU'))
        CompACT = st.sidebar.slider('CompACT', 0.0,32.6,13.9)
        HStile = st.sidebar.slider('HStile', 0,100,17)
        HSGPA = st.sidebar.slider('HSGPA', 0.0,5.164,2.345)
        ActionDate = st.sidebar.slider('ActionDate', 1,1115,300)
        ApplDate = st.sidebar.slider('ApplDate', 1,401,207)
        BDt = st.sidebar.slider('BDt', 1,11996,4207)
        TotalSAT = st.sidebar.slider('TotalSAT', 0,2745,1207)
        zipdisk = st.sidebar.slider('zipdisk', 0.0,18522.5,11420.0)
        data = {'Gender': Gender,
                'Ethnicity': Ethnicity,
                'AcadPlan': AcadPlan,
                'CompACT': CompACT,
                'HStile': HStile,
                'HSGPA': HSGPA,
                'ActionDate': ActionDate,
                'ApplDate': ApplDate,
                'BDt': BDt,
                'TotalSAT': TotalSAT,
                'zipdisk': zipdisk}
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()
    nbofRow = 1

# Combines user input features with entire rose dataset
# This will be useful for the encoding phase
rose_raw = pd.read_csv('rosedata.csv')
rose = rose_raw.drop(columns=['Enroll18'])
df = pd.concat([input_df,rose],axis=0)
#df= input_df

# Encoding of ordinal features
encode = ['Gender','Ethnicity','AcadPlan']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df[:nbofRow] # Selects only the first row (the user input data)

# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(input_df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(input_df)

# Reads in saved classification model
load_clf = pickle.load(open('rose_clf2.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)
dfresult1 = pd.DataFrame(prediction, columns = ['Enroll prediction'])
dfresult2 = pd.DataFrame(prediction_proba, columns = ['Prob Not Enroll','Prob Enroll'])
result = pd.concat([input_df,dfresult1,dfresult2],axis=1)
result.to_csv('result.csv')


st.subheader('Prediction')
Enroll_result = np.array(['notEnroll','Enroll'])
st.write(Enroll_result[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)

st.subheader('Final results')
st.write(result)