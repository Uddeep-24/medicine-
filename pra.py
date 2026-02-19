import streamlit as st
import google.generativeai as genai
from PIL import Image
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini
# API_KEY = os.getenv("GEMINI_API_KEY")
# if API_KEY:
#     genai.configure(api_key=API_KEY)
# else:
#     st.error("Gemini API Key not found. Please set it in your environment variables.")

st.set_page_config(
    page_title="AI Prescription Decoder",
    page_icon="💊",
    layout="wide"
)

# Custom CSS for Premium Look
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        background-color: #007bff;
        color: white;
        font-weight: bold;
    }
    .header-text {
        color: #1a2a6c;
        text-align: center;
        font-family: 'Inter', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

def analyze_prescription(image):
    """
    Analyzes the prescription image using available Gemini models with fallback.
    """
    # Priority list of models to try
    models_to_try = ['gemini-1.5-flash', 'gemini-2.0-flash', 'gemini-1.5-flash-latest', 'gemini-flash-latest']
    
    last_error = ""
    
    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name)
            
            prompt = """
            You are a highly skilled medical pharmacist. 
            Analyze the provided image of a handwritten prescription.
            1. Transcribe the handwriting accurately.
            2. Extract the following details for each medicine:
               - Medicine Name
               - Dosage (e.g., 500mg, 10ml)
               - Frequency (e.g., Twice a day, 1-0-1)
               - Duration (e.g., 5 days)
               - Special Instructions (e.g., Before food, avoid dairy)
            3. Provide a brief summary of what the prescription is for (if discernible).
            4. ADD A STRONG DISCLAIMER: "This is an AI-generated interpretation. Please verify with a qualified pharmacist or doctor."
            
            Format the output in a clean, structured way using Markdown tables.
            """
            
            response = model.generate_content([prompt, image])
            return f"*(Using model: {model_name})*\n\n" + response.text
            
        except Exception as e:
            last_error = str(e)
            # If it's a quota error or 404, try the next model
            if "429" in last_error or "404" in last_error:
                continue
            else:
                return f"Error during analysis: {last_error}"
    
    return f"Failed to analyze. Last error: {last_error}\n\nAll tried models: {models_to_try}"

def main():
    st.markdown("<h1 class='header-text'>🩺 AI Prescription Decoder</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Transform unreadable handwriting into clear medical guidance.</p>", unsafe_allow_html=True)
    
    st.divider()
    
    # Simple sidebar for API key if not in Env
    with st.sidebar:
        st.title("Settings")
        api_key = st.text_input("Enter Gemini API Key", type="password")
        if api_key:
            genai.configure(api_key=api_key)
            st.success("API Key Configured!")
        else:
            st.warning("Please enter your Gemini API Key to proceed.")

    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.subheader("📤 Upload Prescription")
        uploaded_file = st.file_uploader("Choose an image of your prescription...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            if st.button("Analyze Prescription"):
                if not api_key:
                    st.error("Please provide an API key in the sidebar first.")
                else:
                    with st.spinner("🧠 AI is decoding the handwriting..."):
                        analysis_result = analyze_prescription(image)
                        st.session_state['analysis_result'] = analysis_result
    
    with col2:
        st.subheader("📋 Clear Interpretation")
        if 'analysis_result' in st.session_state:
            st.markdown(st.session_state['analysis_result'])
        else:
            st.info("Upload and analyze a prescription to see the digital interpretation here.")

if __name__ == "__main__":
    main()
