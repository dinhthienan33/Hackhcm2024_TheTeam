import streamlit as st
# from transformers import DetrImageProcessor, DetrForObjectDetection
import easyocr
import numpy as np
from PIL import Image
import requests
import json
from transformers import pipeline
from groq import Groq
#from fuzzywuzzy import process

# Initialize the OCR reader

def get_image_caption(image):
    result=[]
    # Use a pre-trained image captioning model from Salesforce
    caption_pipeline_1 = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    result.append(caption_pipeline_1(image)[0]['generated_text'])
    caption_pipeline_2 = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning") 
    result.append(caption_pipeline_2(image)[0]['generated_text'])
    return result

def perform_ocr(image):
    ocr_reader = easyocr.Reader(['en'],gpu=False)
    result = ocr_reader.readtext(np.array(image))
    ocr_texts = [line[1] for line in result]
    if(ocr_texts):
        return ocr_texts
    else: return ["Nothing"]
# def perform_object(image):
#     processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
#     model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
#     inputs = processor(images=image, return_tensors="pt")
#     outputs = model(**inputs)
#     result=outputs["labels"]
#     return result
# def correct_text(ocr_texts):
#     corrected_text = []
#     known_terms = ["Nothing",'Tiger', 'Pepsi', 'Heineken', 'Larue','Bivina','Edelweiss','Bia Viet','Strongbow','Beer carton','Beer crate','Beer bottle','Beer can','Drinker','Promotion Girl','Seller','Buyer','Customer','Ice bucket', 'Ice box', 'Fridge', 'Signage', 'billboard', 'poster', 'standee', 'Tent card', 'display stand', 'tabletop', 'Parasol']
#     for text in ocr_texts:
#         match, score = process.extractOne(text, known_terms)
#         if score > 80:
#             corrected_text.append(match)
#     return corrected_text

def analyze_image_information(image_description, ocr_results):
    prompt = f"""
    Analyze the following image information and provide insights based on the criteria given below:

    Image Description:
    {image_description}

    OCR_result:
    {ocr_results}
    
    Imagine you are a member of the Digital & Technology (D&T) team at HEINEKEN Vietnam. Develop an image analysis tool that can automatically detect the following elements based on Description and OCR_result:
    Just focus on result of OCR that similar to brands
    Criteria:
    1.Brand Logos: Identify any brand logos mentioned in OCR results that match the known brands.
    2.Products: Mention any products such as beer kegs and bottles in the description or OCR results that match the known objects. If OCR results do not contain any objects, do not mention this criterion.
    3.Customers: Describe the number of customers, their activities, and their emotions if any are present.
    4.Promotional Materials: Identify any promotional materials such as posters, banners, and billboards.
    5.Setup Context: Determine the scene context (e.g., bar, restaurant, grocery store, or supermarket).
    Please give me your best result
    Insights:
    Summarize all of criteria and give context
    """

    # Replace with your Groq API key

    client = Groq(
        # This is the default and can be omitted
        api_key="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",
    )


    data = {
        "model": "llama3-8b-8192",
        "messages": [{"role": "user", "content": prompt}],
    }

    chat_completion = client.chat.completions.create(**data)
    return chat_completion.choices[0].message.content


# Streamlit app
logo_path = "img/logo.png"  # Change this to the path of your logo file or a URL
logo = Image.open(logo_path)
st.set_page_config(layout="wide")

st.image(logo, width=100)
custom_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');

h1,h2,h3,h4,h5,h6 {
    font-family: 'Roboto', sans-serif;
}
</style>
"""

# Display the custom CSS in Streamlit
st.markdown(custom_css, unsafe_allow_html=True)

st.markdown("<h1 style='font-size: 48px; color:#80091d; text-align:center;'>The Team</h1>", unsafe_allow_html=True)     
st.markdown("<h1 style='font-size: 30px; text-align:center;'>Image Analysis App</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='font-size: 30px; text-align:center;'>Hack HCM2024</h1>", unsafe_allow_html=True)
# Create three columns with custom widths
# col1, col2, col3 = st.columns([1, 2, 2])

# with col1:
#     st.header("Upload Image")
#     uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

# with col2:
#     st.header("OCR and Description")

#     if uploaded_file is not None:
#         # Load the image
#         image = Image.open(uploaded_file).convert("RGB")
#         st.image(image, caption='Uploaded Image', use_column_width=True)

#         # Get image caption
#         st.subheader("Image Description")
#         image_description = get_image_caption(image)
#         image_description_last=' '.join(image_description)
#         st.write(image_description)

#         # # Perform OCR
#         # st.subheader("OCR Texts")
#         ocr_texts = perform_ocr(image)
#         # object_names = perform_object(image)
#         #ocr_texts=correct_text(ocr_texts)
#         # for text in ocr_texts:
#         #     st.write(text)

# with col3:
#     st.header("Analysis")

#     if uploaded_file is not None:
#         # Analyze image information
#         ocr_results = ' '.join(ocr_texts)
#         # object_names =' '.join(object_names)
#         analysis = analyze_image_information(image_description_last, ocr_results)
#         st.write(analysis)

st.header("Upload Image")
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Get image caption
    st.header("Image Description")
    image_description = get_image_caption(image)
    st.write(image_description)

    # Section 2: OCR and Description
    #st.header("OCR Texts")
    #ocr_texts_pre = perform_ocr(image)
    ocr_texts = perform_ocr(image)
    # for text in ocr_texts:
    #     st.write(text)

    # Section 3: Analysis
    st.header("Analysis")
    # Analyze image information
    ocr_results = ' '.join(ocr_texts) if ocr_texts else ""
    analysis = analyze_image_information(image_description, ocr_results)
    st.write(analysis)