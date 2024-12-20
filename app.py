import streamlit as st
import os
import base64
from together import Together
from dotenv import load_dotenv
import tempfile
from PIL import Image
import subprocess
import sys

# Load environment variables
load_dotenv()
TOGETHER_API_KEY = os.environ.get('TOGETHER_API_KEY')

# Initialize Together client
client = Together()

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Function to analyze image using Llama
def get_llama(image_path):
    base64_image = encode_image(image_path)
    
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

# Streamlit UI
def main():
    st.title("Image Analysis")
    st.write("Upload an image to get AI-powered analysis")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Add an analyze button
        if st.button("Analyze Image"):
            with st.spinner("Analyzing image..."):
                # Save the uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_path = tmp_file.name

                try:
                    # Get analysis from Llama
                    analysis = get_llama(temp_path)
                    
                    # Display results
                    st.subheader("Analysis Results")
                    st.write(analysis)
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                
                finally:
                    # Clean up temporary file
                    os.unlink(temp_path)
