import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import tempfile
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import keras.utils as keras_utils
import io
import pydicom
import nibabel as nib
import zipfile
from pathlib import Path
import cv2
import time
from fpdf import FPDF
from hashlib import sha256

# =============================================
# MODEL LOADING (OPTIMIZED FOR STREAMLIT SHARING)
# =============================================

import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the saved model with caching and error handling
@st.cache_resource
def load_model_cached():
    try:
        # Define custom normalization layer
        class CustomNormalization(tf.keras.layers.Layer):
            def __init__(self, **kwargs):
                super(CustomNormalization, self).__init__(**kwargs)

            def build(self, input_shape):
                self.mean = self.add_weight(name='mean',
                                            shape=(input_shape[-1],),
                                            initializer='zeros',
                                            trainable=False)
                self.variance = self.add_weight(name='variance',
                                                shape=(input_shape[-1],),
                                                initializer='ones', 
                                                trainable=False)
                self.count = self.add_weight(name='count',
                                             shape=(),
                                             initializer='zeros',
                                             trainable=False)
                super(CustomNormalization, self).build(input_shape)

            def call(self, inputs):
                return tf.keras.backend.in_train_phase(
                    tf.nn.batch_normalization(inputs, self.mean, self.variance, None, None, 1e-3),
                    inputs
                )

        # Load model from local directory (no more GitHub download)
        model_path = "models/brain_tumor_classification_model_32pts.h5"  # Ensure this path is correct
        
        # Load model with custom objects
        model = load_model(model_path, 
                           custom_objects={'CustomNormalization': CustomNormalization},
                           compile=False)

        # Recompile model
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model

    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.stop()

# Load the model once when the app starts
try:
    model = load_model_cached()
except Exception as e:
    st.error(f"Critical error loading model: {str(e)}")
    st.stop()


# =============================================
# AUTHENTICATION SYSTEM (SIMPLIFIED FOR DEMO)
# =============================================

# User credentials (for demo purposes only)
USER_CREDENTIALS = {
    "admin": {
        "password": sha256("admin123".encode()).hexdigest(),
        "name": "Administrator",
        "role": "admin"
    },
    "doctor": {
        "password": sha256("doctor123".encode()).hexdigest(),
        "name": "Medical Doctor",
        "role": "physician"
    }
}

# Custom CSS for styling
def set_custom_css():
    st.markdown("""
        <style>
            /* Simplified CSS for better performance */
            .login-container {
                background-color: #f5f5f5;
                padding: 2rem;
                border-radius: 10px;
                max-width: 500px;
                margin: 0 auto;
            }
            
            h1, h2, h3 {
                color: #2E86AB !important;
            }
            
            .stButton>button {
                background-color: #2E86AB !important;
                color: white !important;
                border-radius: 5px;
            }
        </style>
    """, unsafe_allow_html=True)

# Authentication function
def authenticate(username, password):
    if username in USER_CREDENTIALS:
        hashed_password = sha256(password.encode()).hexdigest()
        if USER_CREDENTIALS[username]["password"] == hashed_password:
            return True
    return False

# Login page
def show_login_page():
    set_custom_css()
    
    st.markdown("""
        <div style='text-align: center; margin-bottom: 2rem;'>
            <h1 style='color: #2E86AB;'>NeuroAI Diagnostics</h1>
            <p>Advanced Brain Tumor Classification System</p>
        </div>
    """, unsafe_allow_html=True)
    
    with st.container():
        st.markdown("<div class='login-container'>", unsafe_allow_html=True)
        
        st.markdown("<h2 style='text-align: center;'>Login</h2>", unsafe_allow_html=True)
        
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login", key="login_button"):
            if authenticate(username, password):
                st.session_state["authenticated"] = True
                st.session_state["username"] = username
                st.session_state["user_info"] = USER_CREDENTIALS[username]
                st.success("Login successful! Redirecting...")
                time.sleep(1)
                st.experimental_rerun()
            else:
                st.error("Invalid username or password")
        
        st.markdown("</div>", unsafe_allow_html=True)

# =============================================
# MAIN APPLICATION (OPTIMIZED FOR STREAMLIT SHARING)
# =============================================

# Define constants
IMG_SIZE = (224, 224)
CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
ALLOWED_FILE_TYPES = ["jpg", "jpeg", "png", "dcm"]  # Removed nii for simplicity
MAX_FILE_SIZE_MB = 5  # Reduced for Streamlit Sharing

# Performance metrics
PERFORMANCE_METRICS = {
    'Glioma': {'sensitivity': 0.94, 'specificity': 0.89, 'accuracy': 0.92},
    'Meningioma': {'sensitivity': 0.91, 'specificity': 0.93, 'accuracy': 0.91},
    'No Tumor': {'sensitivity': 0.96, 'specificity': 0.95, 'accuracy': 0.95},
    'Pituitary': {'sensitivity': 0.90, 'specificity': 0.92, 'accuracy': 0.91},
}

# Helper functions with error handling
def get_img_array(img_bytes, size=IMG_SIZE):
    try:
        img = keras_utils.load_img(io.BytesIO(img_bytes), target_size=size)
        array = keras_utils.img_to_array(img)
        return np.expand_dims(array, axis=0)
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

def preprocess_dicom(file_bytes):
    try:
        dicom_data = pydicom.dcmread(io.BytesIO(file_bytes))
        img = dicom_data.pixel_array
        img = cv2.resize(img, IMG_SIZE)
        img = np.expand_dims(img, axis=-1)
        img = np.repeat(img, 3, axis=-1)
        img = img.astype(np.float32) / 255.0
        return np.expand_dims(img, axis=0)
    except Exception as e:
        st.error(f"Error processing DICOM file: {str(e)}")
        return None

def make_gradcam_heatmap(img_array, model=model, last_conv_layer_name="Top_Conv_Layer"):
    try:
        grad_model = tf.keras.models.Model(
            model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
        )
        
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]
            
        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
    except Exception as e:
        st.error(f"Error generating heatmap: {str(e)}")
        return None

def main_app():
    set_custom_css()
    
    # Sidebar with user info
    with st.sidebar:
        st.markdown(f"""
            <div style='background-color: #2E86AB; color: white; padding: 0.5rem; border-radius: 5px;'>
                <p style='margin: 0;'>Logged in as: <strong>{st.session_state.user_info['name']}</strong></p>
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("Logout"):
            st.session_state.clear()
            st.experimental_rerun()
    
    # Main application content
    st.title("🧠 NeuroAI Brain Tumor Diagnostics")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload an MRI Image", 
        type=ALLOWED_FILE_TYPES,
        help=f"Supported formats: {', '.join(ALLOWED_FILE_TYPES)}"
    )
    
    # Initialize session state
    if "uploaded_image" not in st.session_state:
        st.session_state.uploaded_image = None
    if "prediction" not in st.session_state:
        st.session_state.prediction = None
    
    # Handle file upload
    if uploaded_file is not None:
        try:
            file_extension = uploaded_file.name.split(".")[-1].lower()
            
            # Check file size
            file_size_mb = uploaded_file.size / (1024 * 1024)
            if file_size_mb > MAX_FILE_SIZE_MB:
                st.error(f"File size exceeds the limit of {MAX_FILE_SIZE_MB} MB.")
            else:
                with st.spinner("Processing image..."):
                    if file_extension == "dcm":
                        img_array = preprocess_dicom(uploaded_file.getvalue())
                    else:
                        img_array = get_img_array(uploaded_file.getvalue())
                    
                    if img_array is not None:
                        st.session_state.uploaded_image = uploaded_file.getvalue()
                        st.session_state.img_array = img_array
                        st.success("Image processed successfully!")
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    # Display uploaded image
    if st.session_state.uploaded_image is not None:
        st.subheader("Uploaded Image")
        st.image(st.session_state.uploaded_image, use_column_width=True)
    
    # Prediction button
    if st.session_state.uploaded_image is not None and st.button("Analyze Image"):
        with st.spinner("Analyzing..."):
            try:
                preds = model.predict(st.session_state.img_array)
                st.session_state.prediction = CLASS_NAMES[np.argmax(preds[0])]
                st.session_state.prediction_probabilities = preds[0]
                
                # Display results
                st.subheader("Analysis Results")
                st.write(f"**Prediction:** {st.session_state.prediction}")
                st.write(f"**Confidence:** {np.max(preds[0]) * 100:.2f}%")
                
                # Show probabilities
                st.write("**Probabilities:**")
                for i, class_name in enumerate(CLASS_NAMES):
                    st.write(f"- {class_name}: {preds[0][i] * 100:.2f}%")
                
                # Performance metrics
                st.write("**Performance Metrics:**")
                metrics = PERFORMANCE_METRICS[st.session_state.prediction]
                st.write(f"- Sensitivity: {metrics['sensitivity'] * 100:.0f}%")
                st.write(f"- Specificity: {metrics['specificity'] * 100:.0f}%")
                st.write(f"- Accuracy: {metrics['accuracy'] * 100:.0f}%")
                
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
    
    # Heatmap generation
    if st.session_state.prediction is not None and st.button("Generate Heatmap"):
        with st.spinner("Generating heatmap..."):
            try:
                heatmap = make_gradcam_heatmap(st.session_state.img_array)
                if heatmap is not None:
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.imshow(heatmap, cmap="jet")
                    plt.axis("off")
                    st.subheader("Activation Heatmap")
                    st.pyplot(fig)
            except Exception as e:
                st.error(f"Error generating heatmap: {str(e)}")

# =============================================
# APP FLOW CONTROL
# =============================================

def main():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        show_login_page()
    else:
        main_app()

if __name__ == "__main__":
    main()