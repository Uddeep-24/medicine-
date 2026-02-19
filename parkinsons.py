import streamlit as st
import pickle
import numpy as np

# 1. Load the Parkinson's model and scaler
try:
    model = pickle.load(open('parkinsons_model.sav', 'rb'))
    scaler = pickle.load(open('parkinsons_scaler.sav', 'rb'))
except FileNotFoundError:
    st.error("Parkinson's Model or Scaler files not found! Please check filenames.")

def main():
    st.set_page_config(page_title="Parkinson's Disease Prediction", layout="wide")
    st.title("Parkinson's Disease Prediction App")
    st.write("Enter the voice acoustic parameters below:")

    # 2. UI Layout - Organized into 3 columns for better visibility
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fo = st.text_input('MDVP:Fo(Hz)', value="0")
        fhi = st.text_input('MDVP:Fhi(Hz)', value="0")
        flo = st.text_input('MDVP:Flo(Hz)', value="0")
        jitter_p = st.text_input('MDVP:Jitter(%)', value="0")
        jitter_abs = st.text_input('MDVP:Jitter(Abs)', value="0")
        rap = st.text_input('MDVP:RAP', value="0")
        ppq = st.text_input('MDVP:PPQ', value="0")
        
    with col2:
        ddp = st.text_input('Jitter:DDP', value="0")
        shimmer = st.text_input('MDVP:Shimmer', value="0")
        shimmer_db = st.text_input('MDVP:Shimmer(dB)', value="0")
        apq3 = st.text_input('Shimmer:APQ3', value="0")
        apq5 = st.text_input('Shimmer:APQ5', value="0")
        apq = st.text_input('MDVP:APQ', value="0")
        dda = st.text_input('Shimmer:DDA', value="0")
        
    with col3:
        nhr = st.text_input('NHR', value="0")
        hnr = st.text_input('HNR', value="0")
        rpde = st.text_input('RPDE', value="0")
        dfa = st.text_input('DFA', value="0")
        spread1 = st.text_input('spread1', value="0")
        spread2 = st.text_input('spread2', value="0")
        d2 = st.text_input('D2', value="0")
        ppe = st.text_input('PPE', value="0")

    # 3. Prediction Logic
    if st.button("Predict Parkinson's Status"):
        # Convert all 22 inputs to floats
        try:
            features = [fo, fhi, flo, jitter_p, jitter_abs, rap, ppq, ddp,
                        shimmer, shimmer_db, apq3, apq5, apq, dda, nhr, hnr,
                        rpde, dfa, spread1, spread2, d2, ppe]
            
            user_input = [float(x) for x in features]
            
            # Standardize and Predict
            input_data = np.asarray(user_input).reshape(1,-1)
            std_data = scaler.transform(input_data)
            prediction = model.predict(std_data)

            if prediction[0] == 1:
                st.error("The model predicts the person has Parkinson's Disease.")
            else:
                st.success("The model predicts the person does NOT have Parkinson's Disease.")
        except ValueError:
            st.warning("Please enter valid numbers in all fields.")

if __name__ == '__main__':
    main()
