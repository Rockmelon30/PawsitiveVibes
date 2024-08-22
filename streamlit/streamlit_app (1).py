
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import base64


def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

img_base64 = get_base64_of_bin_file('zoro.jpg')
side_img_base64 = get_base64_of_bin_file('paw background.jpg')

# setting my dog as the background
page_bg_img = f'''
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/png;base64,{img_base64}");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: scroll;
}}
</style>
'''

# setting sidebar background
sidebar_bg_css = '''
<style>
[data-testid="stSidebar"] {{
    background-color: rgba(0, 0, 0, 0.8); /* Black background with 80% opacity */
    background-image: url("data:image/png;base64,{side_img_base64}"); /* Optional: add an image */
    background-size: cover; /* Ensure the background image covers the sidebar */
    background-repeat: no-repeat; /* Prevent the background image from repeating */
    color: white; /* Set the text color to white for visibility */
}}
</style>
'''

# for opaque headers/titles for better visibility
header_css = '''
<style>
.header-box {
    background-color: rgba(0, 0, 0, 0.7); /* Black background with 70% opacity */
    color: white;                        /* White text color */
    padding: 20px;                       /* Padding around the text */
    border-radius: 10px;                 /* Rounded corners */
    margin-bottom: 20px;                 /* Space below the box */
    text-align: center;                  /* Center the text (optional) */
}
</style>
'''

# for opaque paragraphs for better visibility
opaque_css = '''
<style>
.paragraph-box {
    background-color: rgba(0, 0, 0, 0.7); /* Black background with 70% opacity */
    color: white;                        /* White text color */
    padding: 15px;                       /* Padding around the text */
    border-radius: 10px;                 /* Rounded corners */
    margin-bottom: 20px;                 /* Space below the box */
}
</style>
'''

# for opaque file uploader box for better visibility
custom_css = '''
<style>
div[data-testid="stFileUpload"] {{
    background-color: rgba(0, 0, 0, 0.7); /* Black background with 70% opacity */
    padding: 15px;                        /* Padding around the uploader */
    border-radius: 10px;                  /* Rounded corners */
    margin-bottom: 20px;                  /* Space below the uploader */
}}

div[data-testid="stFileUpload"] label {{
    color: white;                        /* White text color */
    display: block;                      /* Make sure the label is block level */
    padding: 10px;                       /* Padding around the label text */
    background-color: rgba(0, 0, 0, 0.7); /* Black background for label */
    border-radius: 5px;                  /* Rounded corners */
}}
</style>
'''

selectbox_css = '''
<style>
div[data-testid="stSidebar"] div[data-testid="stSelectbox"] > div:first-child {{
    background-color: rgba(0, 0, 0, 0.7);
    color: white;
    padding: 10px;
    border-radius: 10px;
}}
div[data-testid="stSidebar"] div[data-testid="stSelectbox"] label {{
    color: white;
}}
</style>
'''

prediction_css = '''
<style>
.prediction-box {
    background-color: rgba(0, 0, 0, 0.7); /* Black background with 70% opacity */
    color: white;                        /* White text color */
    padding: 10px;                       /* Padding around the text */
    border-radius: 10px;                 /* Rounded corners */
    margin-top: 20px;                    /* Space above the prediction box */
    margin-bottom: 20px;                 /* Space below the prediction box */
    text-align: center;                  /* Center the text */
}
</style>
'''

st.markdown(prediction_css, unsafe_allow_html=True)
st.markdown(sidebar_bg_css.format(side_img_base64=side_img_base64), unsafe_allow_html=True)
st.markdown(custom_css, unsafe_allow_html=True)
st.markdown(opaque_css, unsafe_allow_html=True)
st.markdown(header_css, unsafe_allow_html=True)
st.markdown(page_bg_img, unsafe_allow_html=True)
st.markdown(selectbox_css, unsafe_allow_html=True)

st.markdown('<h1 class="header-box">&#x1F43E; Pawsitive Vibes &#x1F43E;</h1>', unsafe_allow_html=True)
st.markdown('<h2 class="header-box">Dog Emotion Image Classifier</h2>', unsafe_allow_html=True)

model = tf.keras.models.load_model('./best_model.keras')

def preprocess_image(uploaded_file):
    target_size=(256, 256)
    img = Image.open(uploaded_file)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(model, img_array):
    prediction = model.predict(img_array)
    return prediction[0]


activities = ['Classify', 'About']
choice = st.sidebar.selectbox('Select Activity', activities)

if choice == 'Classify':
    st.markdown('<h3 class="header-box">Emotion Classification App</h3>', unsafe_allow_html=True)

    image_file = st.file_uploader('Upload Image', type=['jpg', 'jpeg', 'png', 'bmp'])

    if image_file is not None:
        try:
            image = Image.open(image_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)

            preprocessed_image = preprocess_image(image_file)

            pred = predict_image(model, preprocessed_image)

            class_labels = ['Angry', 'Happy', 'Sad', 'Relaxed']
            pred_class = class_labels[np.argmax(pred)]

            st.markdown(f'<div class="prediction-box">Prediction: {pred_class}</div>', unsafe_allow_html=True)
            print('wrote output')
        except Exception as e:
            st.error(f"An error has occurred: {e}")

elif choice == 'About':
    st.markdown('<h3 class="header-box">About</h3>', unsafe_allow_html=True)
    st.markdown('<div class="paragraph-box">This app uses a CNN model to classify emotions from uploaded images of dog faces. The model makes use of the pre-trained VGG16 model and ImageNet for weights. The model was trained on a total of 4000 photos of dog faces, separated into four distinct and balanced classes of Angry, Happy, Sad and Relaxed. Upload a photo of your pet and see if my model can accurately classify their emotions!<br><br> This model performs with 86 percent accuracy.<br><br> Photo credit goes to Zoro the Rottie.</div>', unsafe_allow_html=True)
    
