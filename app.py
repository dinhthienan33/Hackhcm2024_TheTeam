import streamlit as st
import numpy as np
from PIL import Image
from transformers import pipeline
from groq import Groq
from paddleocr import PaddleOCR
from fuzzywuzzy import process
# Initialize OCR reader

# Perform OCR on image
def perform_ocr(image):
    ocr_reader = PaddleOCR(use_angle_cls=True, lang='en')
    result = ocr_reader.ocr(np.array(image))
    ocr_texts = [line[1][0] for line in result]
    return ocr_texts

def correct_text(ocr_texts):
    corrected_text = []
    known_terms = ['Tiger', 'Pepsi', 'Heineken', 'Larue','Bivina','Edelweiss','Bia Viet','Strongbow','Beer carton','Beer crate','Beer bottle','Beer can','Drinker','Promotion Girl','Seller','Buyer','Customer','Ice bucket', 'Ice box', 'Fridge', 'Signage', 'billboard', 'poster', 'standee', 'Tent card', 'display stand', 'tabletop', 'Parasol']
    for text in ocr_texts:
        match, score = process.extractOne(text, known_terms)
        if score > 50:
            corrected_text.append(match)
        else:
            corrected_text.append(text)
    return corrected_text

# Get image description using image captioning model
def get_image_caption(image):
    caption_pipeline = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    return caption_pipeline(image)[0]['generated_text']

# Analyze image information using Groq API
def analyze_image_information(image_description, ocr_results):
    prompt = f"""
    Analyze the following image information and provide insights based on the criteria given below:

    Image Description:
    {image_description}

    OCR Results:
    {ocr_results}

    Criteria:
    1. Brand Logos: Identify any brand logos mentioned in the description or OCR results.
    2. Products: Mention any products such as beer kegs and bottles in the description or OCR results.
    3. Customers: Describe the number of customers, their activities, and emotions.
    4. Promotional Materials: Identify any posters, banners, and billboards.
    5. Setup Context: Determine the scene context (e.g., bar, restaurant, grocery store, or supermarket).

    Insights:
    Summarize all of criteria and give context
    """

    # Replace with your Groq API key
    client = Groq(api_key="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")

    data = {
        "model": "llama3-8b-8192",
        "messages": [{"role": "user", "content": prompt}]
    }

    chat_completion = client.chat.completions.create(**data)
    return chat_completion.choices[0].message.content

# Streamlit app
st.set_page_config(layout="wide")
st.title("Image Analysis App")

# Create three columns with custom widths
col1, col2, col3 = st.columns([1, 2, 2])

with col1:
    st.header("Upload Image")
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

with col2:
    st.header("OCR and Description")

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='Uploaded Image', use_column_width=True)

        st.subheader("Image Description")
        image_description = get_image_caption(image)
        st.write(image_description)

        ocr_texts = perform_ocr(image)
        ocr_texts = correct_text(ocr_texts)
        st.subheader("OCR Texts")
        for text in ocr_texts:
            st.write(text)

with col3:
    st.header("Analysis")

    if uploaded_file is not None:
        ocr_results = ' '.join(ocr_texts)
        analysis = analyze_image_information(image_description, ocr_results)
        st.write(analysis)