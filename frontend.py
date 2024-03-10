import numpy as np
import pandas as pd
import pickle
import streamlit as st

xgb_model = pickle.load(open('/xgb_model.sav', 'rb'))


def main():

# Apply the style to the title
    st.title('Obesity Classification Predictor')
    st.markdown('<p style="color: white;">Please answer the questions below and click the get results button to get your classification.</p>', unsafe_allow_html=True)
    st.markdown('Do **NOT** click the enter button until all of the questions are answered, as the enter button submits the form')

    with st.form(key="myform", clear_on_submit=True):
        gender = st.selectbox("What is your gender?", ("Female", "Male"), placeholder="Enter your gender", index=None, key="gender")
        age = st.number_input("What is your age?", value=None, placeholder="Enter your age", min_value=0, key="age")
        height = st.number_input("What is your height in inches?", value=None, placeholder="Enter your height", min_value=0.0, key="height") #convert to meters
        weight = st.number_input("What is your weight in pounds?", value=None, placeholder="Enter your weight", min_value=0.0, key="weight") #convert to kg
        family_history = st.selectbox("Has a family member suffered or suffers from overweight?",("no", "yes"), index=None, key="family_history")
        FAVC = st.selectbox("Do you eat high caloric food frequently?", ("no", "yes"), index=None, key="FAVC") 
        FCVC = st.selectbox("Do you usually eat vegetables in your meals?", ("never", "sometimes", "always"), index=None, key="FCVC") #convert to 1,2,3
        if FCVC == "never":
            fcvc = 1
        elif FCVC == "sometimes":
            fcvc = 2
        else:
            fcvc = 3
        NCP = st.selectbox("How many main meals do you have daily",("1 to 2 meals", "3 meals", "More than 3"), index=None, key ="NCP")
        if NCP == "1 to 2 meals":
            ncp = 1
        elif NCP == "3 meals":
            ncp = 3
        else:
            ncp = 4
        CAEC = st.selectbox("Do you eat any food between meals?", ("no", "Sometimes", "Frequently", "Always"), index=None, key ="CAEC")
        SMOKE = st.selectbox("Do you smoke?", ("no", "yes"), index=None, key ="SMOKE")
        CH2O = st.selectbox("How much water do you drink daily?", ("Less than a liter", "Between 1 and 2 Liters", "More than 2L"), index=None, placeholder="There are around 4 cups of water in a Liter", key ="CH2O")#convert to 1,2,3
        if CH2O == "Less than a liter":
            ch2o = 1
        elif CH2O == "Between 1 and 2 Liters":
            ch2o = 2
        else:
            ch2o = 3
        SCC = st.selectbox("Do you monitor the calories you eat daily?", ("no", "yes"), index=None, key ="SCC")
        FAF = st.selectbox("How often do you do physical activity a week?", ("Never", "1 to 2 days", "2 to 4 days", "More than 4 days"), index=None, key="FAF") # convert to 0, 1, 2, 3
        if FAF == "Never":
            faf = 0
        elif FAF == "1 to 2 days":
            faf = 1
        elif FAF == "2 to 4 days":
            faf = 2
        else:
            faf = 3
        TUE = st.selectbox("How often are you on technology devices a day?", ("0 to 2 hours", "3 to 5 hours", "More than 5 hours"),index=None, key="TUE") # convert to 0, 1, 2
        if TUE == "0 to 2 hours":
            tue = 0
        elif TUE == "3 to 5 hours":
            tue = 1
        else:
            tue = 2
        CALC = st.selectbox("How often do you drink alcohol?", ("no", "Sometimes", "Frequently", "Always"), index=None, key="CALC")
        MTRANS = st.selectbox("Which mode of transportation do you usually use?", ("Automobile", "Motorbike", "Bike","Public_Transportation", "Walking"), index=None, key="MTRANS")
        submit = st.form_submit_button("Get Results")
    obesity_classification = ''
    
    if submit:
        obesity_classification = give_prediction([gender, age, (height*0.0254), (weight/2.205), family_history, FAVC, fcvc, ncp, CAEC, SMOKE, ch2o,SCC, faf, tue, CALC, MTRANS])
    
    st.success(obesity_classification)


def give_prediction(input_data):

    user_data = pd.DataFrame({
        'Gender': [input_data[0]],
        'Age': [input_data[1]],
        'Height': [input_data[2]],
        'Weight': [input_data[3]],
        'family_history_with_overweight': [input_data[4]],
        'FAVC': [input_data[5]],
        'FCVC': [input_data[6]],
        'NCP': [input_data[7]],
        'CAEC': [input_data[8]],
        'SMOKE': [input_data[9]],
        'CH2O': [input_data[10]],
        'SCC': [input_data[11]],
        'FAF': [input_data[12]],
        'TUE': [input_data[13]],
        'CALC': [input_data[14]],
        'MTRANS': [input_data[15]]
    })
    columns = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
    for c in columns:
        dummies = pd.get_dummies(user_data[c], prefix=(str(c)+"_"))
        user_data= pd.concat([user_data, dummies], axis=1)
    user_data = user_data.drop(columns=columns)
    expected = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE',
       'Gender__Female', 'Gender__Male', 'family_history_with_overweight__no',
       'family_history_with_overweight__yes', 'FAVC__no', 'FAVC__yes',
       'CAEC__Always', 'CAEC__Frequently', 'CAEC__Sometimes', 'CAEC__no',
       'SMOKE__no', 'SMOKE__yes', 'SCC__no', 'SCC__yes', 'CALC__Always',
       'CALC__Frequently', 'CALC__Sometimes', 'CALC__no', 'MTRANS__Automobile',
       'MTRANS__Bike', 'MTRANS__Motorbike', 'MTRANS__Public_Transportation',
       'MTRANS__Walking']
    user_data = user_data.reindex(columns=expected)
    user_data = user_data.fillna(0)
    classificationX = xgb_model.predict(user_data)
    x = ''
    if classificationX == 0:
        x = "XGB Model: Insufficient Weight"
    elif classificationX == 1:
        x ='XGB Model: Normal Weight'
    elif classificationX == 2:
        x = 'XGB Model: Overweight Level I'
    elif classificationX == 3:
        x = 'XGB Model: Overweight Level II'
    elif classificationX == 4:
        x = 'XGB Model: Obesity Type I'
    elif classificationX == 5:
        x = 'XGB Model: Obesity Type II'
    elif classificationX == 6:
       x = "XGB Model: Obesity Type III"
    return x

if __name__ == '__main__':
    main()
