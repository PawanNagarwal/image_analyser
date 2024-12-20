import streamlit as st
import os
import base64
from together import Together
from dotenv import load_dotenv

# Load environment variables and setup
load_dotenv()
TOGETHER_API_KEY = os.environ.get('TOGETHER_API_KEY')
client = Together()

# Page title
st.title("Image Analysis App")
st.write("Upload an image to analyze it")

def analyze_image(uploaded_file):
    # Convert uploaded image to base64
    base64_image = base64.b64encode(uploaded_file.getvalue()).decode('utf-8')
    
    # Get analysis from Llama
    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "You will be given an image, tell me the details about that image"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error analyzing image: {str(e)}"

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image")
    
    # Analyze button
    if st.button("Analyze Image"):
        with st.spinner("Analyzing..."):
            # Get and display analysis
            result = analyze_image(uploaded_file)
            st.write("### Analysis Result:")
            st.write(result)

# API key status
if not TOGETHER_API_KEY:
    st.error("Please set your TOGETHER_API_KEY environment variable.")
