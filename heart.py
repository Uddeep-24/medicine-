import streamlit as st
import pickle
import numpy as np

# 1. MUST be the first command
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

# 2. Load the Heart Disease model and scaler
# Ensure these filenames match the files in your C:\ml\ folder
try:
    model = pickle.load(open('heart_model.sav', 'rb'))
    scaler = pickle.load(open('heart_scaler.sav', 'rb'))
    ready = True
except FileNotFoundError:
    st.error("❌ Heart Disease model or scaler files not found in the folder!")
    ready = False

def main():
    st.title("❤️ Heart Disease Prediction App")
    
    if not ready:
        return

    st.write("Enter the following clinical data to predict heart disease risk:")

    # 3. UI Layout - Organized into 2 columns for the 13 features
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input('Age', min_value=1, max_value=120, value=25)
        sex = st.selectbox('Sex (1 = Male, 0 = Female)', options=[1, 0])
        cp = st.selectbox('Chest Pain Type (0, 1, 2, 3)', options=[0, 1, 2, 3])
        trestbps = st.number_input('Resting Blood Pressure (mm Hg)', value=120)
        chol = st.number_input('Serum Cholestoral (mg/dl)', value=200)
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)', options=[1, 0])
        
    with col2:
        restecg = st.selectbox('Resting ECG Results (0, 1, 2)', options=[0, 1, 2])
        thalach = st.number_input('Maximum Heart Rate Achieved', value=150)
        exang = st.selectbox('Exercise Induced Angina (1 = Yes, 0 = No)', options=[1, 0])
        oldpeak = st.number_input('ST depression (Oldpeak)', value=0.0, step=0.1)
        slope = st.selectbox('Slope of the Peak Exercise ST Segment (0, 1, 2)', options=[0, 1, 2])
        ca = st.selectbox('Number of Major Vessels (0-3)', options=[0, 1, 2, 3])
        thal = st.selectbox('Thal (0 = Normal, 1 = Fixed, 2 = Reversable)', options=[0, 1, 2])

    # 4. Prediction Logic
    if st.button("Predict Heart Condition"):
        try:
            # Order MUST match the training dataset
            features = [age, sex, cp, trestbps, chol, fbs, restecg, 
                        thalach, exang, oldpeak, slope, ca, thal]
            
            # Convert to numpy array and reshape
            input_data = np.asarray(features).reshape(1, -1)
            
            # Standardize using your loaded scaler
            std_data = scaler.transform(input_data)
            
            prediction = model.predict(std_data)

            st.markdown("---")
            if prediction[0] == 1:
                st.error("### Result: High Risk of Heart Disease")
            else:
                st.success("### Result: Low Risk of Heart Disease")
                
        except Exception as e:
            st.warning(f"Error during prediction: {e}")

if __name__ == '__main__':
    main()
