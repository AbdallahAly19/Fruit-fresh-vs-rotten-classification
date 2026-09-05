import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# --- Page setup ---
st.set_page_config(page_title="Fruit & Vegetable Quality Checker", page_icon="🍎")
st.title("🍎 Fruit & Vegetable Quality Checker")
st.write("Upload a photo of a fruit or vegetable, and the model will predict whether it's fresh or rotten.")

IMAGE_SIZE = (128, 128)

# --- Load model once, and cache it ---
# @st.cache_resource makes sure the model is loaded only ONCE, the first time
# the app runs, instead of reloading it on every single interaction (which would
# be slow, since loading a deep learning model takes a couple of seconds).
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("fruit_quality_model.keras")

model = load_model()

# Binary model: index 0 = fresh, index 1 = rotten (matches how labels were
# assigned during training in the notebook: 0 for fresh, 1 for rotten)
class_names = ['fresh', 'rotten']

# --- Preprocessing function ---
# This must match EXACTLY how images were preprocessed during training:
# same color conversion, same resize size, same normalization (0-255 -> 0-1).
# If any of these differ from training, predictions will be unreliable.
def preprocess_image(pil_image):
    image = np.array(pil_image.convert("RGB"))  # ensure 3 color channels
    image = cv2.resize(image, IMAGE_SIZE)
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)  # model expects a batch dimension
    return image

# --- File uploader ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file)
    st.image(pil_image, caption="Uploaded image", use_container_width=True)

    if st.button("Predict"):
        with st.spinner("Analyzing..."):
            processed = preprocess_image(pil_image)
            prediction = model.predict(processed)
            predicted_index = int(np.argmax(prediction))
            predicted_label = class_names[predicted_index]
            confidence = float(np.max(prediction))

        if predicted_label == "fresh":
            st.success(f"✅ This looks **fresh** ({confidence:.1%} confidence)")
        else:
            st.error(f"⚠️ This looks **rotten** ({confidence:.1%} confidence)")

        # Show both class probabilities, for transparency
        with st.expander("See full prediction breakdown"):
            probs = prediction[0]
            for idx in range(len(class_names)):
                st.write(f"{class_names[idx]}: {probs[idx]:.1%}")
else:
    st.info("Upload an image to get a prediction.")
