import streamlit as st
import pickle
import numpy as np

# 1. Load your saved model and scaler
# Make sure these filenames match exactly what you saved earlier
try:
    model = pickle.load(open('trained_model.sav', 'rb'))
    scaler = pickle.load(open('scaler.sav', 'rb'))
except FileNotFoundError:
    st.error("Model or Scaler files not found! Please run the saving cells first.")

def main():
    st.title("Diabetes Prediction App")
    st.write("Enter the following details to check the diabetes status:")

    # 2. UI Layout - Adjust these labels to match your dataset columns
    col1, col2 = st.columns(2)
    
    with col1:
        preg = st.text_input('Number of Pregnancies', value="0")
        gluc = st.text_input('Glucose Level', value="0")
        bp = st.text_input('Blood Pressure value', value="0")
        skin = st.text_input('Skin Thickness value', value="0")
        
    with col2:
        ins = st.text_input('Insulin Level', value="0")
        bmi = st.text_input('BMI value', value="0")
        dpf = st.text_input('Diabetes Pedigree Function', value="0")
        age = st.text_input('Age', value="0")

    # 3. Prediction Logic
    if st.button("Predict Result"):
        # Convert inputs to a list of floats
        user_input = [float(preg), float(gluc), float(bp), float(skin), 
                      float(ins), float(bmi), float(dpf), float(age)]
        
        # Standardize and Predict
        input_data = np.asarray(user_input).reshape(1,-1)
        std_data = scaler.transform(input_data)
        prediction = model.predict(std_data)

        if prediction[0] == 1:
            st.error("The result is Positive: The person is likely Diabetic.")
        else:
            st.success("The result is Negative: The person is likely Not Diabetic.")

if __name__ == '__main__':
    main()
