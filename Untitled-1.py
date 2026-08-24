# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PIL import Image
import io
import cv2
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Indian Currency Detector",
    page_icon="🎓",
    layout="wide"
)

# App title and header
st.title("INDIAN CURRENCY AUTHENTICATION SYSTEM")
st.markdown("### Detect genuine vs counterfeit Indian Rupee notes")

# Sidebar information
st.sidebar.header("About Indian Currency")
st.sidebar.markdown("""
## Current Indian Banknotes
- ₹10 (Brown)
- ₹20 (Greenish-Yellow)
- ₹50 (Fluorescent Blue)
- ₹100 (Lavender)
- ₹200 (Bright Yellow)
- ₹500 (Stone Grey)
""")

st.sidebar.header("Security Features")
st.sidebar.markdown("""
1. Watermark
2. Security thread
3. Micro lettering
4. Color-shifting ink
5. Identification mark
6. Latent image
7. Intaglio printing
""")

# Generate synthetic data for Indian currency
@st.cache_data
def generate_indian_currency_data():
    np.random.seed(42)
    n_samples = 2000

    feature_names = [
        'watermark', 'security_thread', 'micro_lettering', 'intaglio',
        'color_shift', 'latent_image'
    ]

    genuine_features = {}
    fake_features = {}

    for feature in feature_names:
        if feature in ['watermark', 'security_thread', 'intaglio', 'color_shift']:
            genuine_features[feature] = np.random.normal(0.95, 0.02, n_samples // 2)
            fake_features[feature] = np.random.normal(0.25, 0.2, n_samples // 2)
        elif feature in ['micro_lettering', 'latent_image']:
            genuine_features[feature] = np.random.normal(0.93, 0.03, n_samples // 2)
            fake_features[feature] = np.random.normal(0.2, 0.15, n_samples // 2)
        else:
            genuine_features[feature] = np.random.normal(0.90, 0.05, n_samples // 2)
            fake_features[feature] = np.random.normal(0.3, 0.2, n_samples // 2)

    all_features = {}
    for feature in feature_names:
        all_features[feature] = np.clip(
            np.concatenate([genuine_features[feature], fake_features[feature]]),
            0, 1
        )

    labels = np.concatenate([np.zeros(n_samples // 2), np.ones(n_samples // 2)])

    data = pd.DataFrame(all_features)
    data['class'] = labels

    return data

data = generate_indian_currency_data()

@st.cache_data
def train_model(data):
    X = data.drop('class', axis=1)
    y = data['class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, accuracy, X.columns.tolist()

model, accuracy, model_feature_names = train_model(data)

st.write(f"Model accuracy on test data: **{accuracy*100:.2f}%**")

# Helper function to add black background to images
def add_black_background(img):
    if isinstance(img, np.ndarray):
        if len(img.shape) == 2:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = img.copy()

        h, w = img.shape[:2]
        black_bg = np.zeros((h + 20, w + 20, 3), dtype=np.uint8)
        black_bg[10:10+h, 10:10+w] = img_rgb

        return Image.fromarray(black_bg)
    else:
        img_array = np.array(img)
        return add_black_background(img_array)

# Enhanced image analysis function
def analyze_currency_image(image):
    # convert to plain RGB first to avoid alpha channels or palettes
    try:
        image = image.convert('RGB')
    except Exception:
        pass
    img_array = np.array(image)
    if img_array.size == 0:
        raise ValueError("Uploaded image appears empty")

    # drop alpha if present
    if img_array.ndim == 3 and img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    h, w = gray.shape[:2] if gray.size > 0 else (100, 100)

    # Edge detection for watermark
    edges = cv2.Canny(gray, 100, 200) if gray.size > 0 else np.zeros((h, w), dtype=np.uint8)
    watermark_score = min(1.0, np.count_nonzero(edges) / max(1, (h * w * 0.2)))

    # Security thread detection
    if gray.size > 0:
        horizontal_sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        security_thread_score = min(1.0, np.std(horizontal_sobel) / 60)
    else:
        security_thread_score = 0.1

    # Micro-lettering detection
    if gray.size > 0:
        _, threshold = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        text_regions = cv2.countNonZero(threshold) / max(1, (h * w))
        micro_lettering_score = min(1.0, text_regions * 3)
    else:
        micro_lettering_score = 0.1
        threshold = np.zeros((h, w), dtype=np.uint8)

    # Intaglio detection
    if gray.size > 0:
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        intaglio_score = min(1.0, np.std(hist) / 6000)
    else:
        intaglio_score = 0.1
        hist = np.zeros((256, 1))

    # Color shift detection
    if len(img_array.shape) == 3 and img_array.size > 0:
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        color_shift_score = min(1.0, np.std(hsv[:,:,0]) / 60)
    else:
        color_shift_score = 0.3

    # Latent image detection
    if gray.size > 0:
        kernel_latent = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        latent_filtered = cv2.filter2D(gray, -1, kernel_latent)
        latent_variance = np.var(latent_filtered) if latent_filtered.size > 0 else 0
        latent_image_score = min(1.0, latent_variance / 6000)
    else:
        latent_image_score = 0.1
        latent_filtered = np.zeros((h, w))

    features = {
        'watermark': min(1.0, max(0.0, watermark_score)),
        'security_thread': min(1.0, max(0.0, security_thread_score)),
        'micro_lettering': min(1.0, max(0.0, micro_lettering_score)),
        'intaglio': min(1.0, max(0.0, intaglio_score)),
        'color_shift': min(1.0, max(0.0, color_shift_score)),
        'latent_image': min(1.0, max(0.0, latent_image_score)),
    }

    # Visualization images
    if gray.size > 0:
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
        magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        hist_img = np.zeros((200, 256, 3), dtype=np.uint8)
        # normalize histogram and flatten to simple list so we can convert elements to ints
        hist_normalized = cv2.normalize(hist, None, 0, 200, cv2.NORM_MINMAX).flatten()
        for i, val in enumerate(hist_normalized):
            cv2.line(hist_img, (i, 200), (i, 200 - int(val)), (255, 255, 255), 1)

        latent_vis = cv2.normalize(latent_filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        latent_vis_rgb = cv2.cvtColor(latent_vis, cv2.COLOR_GRAY2RGB)

        horizontal_sobel_vis = cv2.normalize(horizontal_sobel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        magnitude_spectrum = np.zeros((h, w), dtype=np.uint8)
        hist_img = np.zeros((200, 256, 3), dtype=np.uint8)
        latent_vis_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        horizontal_sobel_vis = np.zeros((h, w), dtype=np.uint8)

    processed_images = {
        'gray': gray if gray.size > 0 else np.zeros((h, w), dtype=np.uint8),
        'edges': edges,
        'threshold': threshold,
        'horizontal_sobel': horizontal_sobel_vis,
        'frequency': magnitude_spectrum,
        'histogram': hist_img,
        'latent': latent_vis_rgb,
    }

    return features, processed_images

# Main app interface
st.header("Upload an image of Indian currency for authentication")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    try:
        input_image = Image.open(uploaded_file)
        st.image(input_image, caption="Uploaded Image")

        with st.spinner("Analyzing currency features..."):
            features, processed_images = analyze_currency_image(input_image)

            X_new = pd.DataFrame([features])[model_feature_names]

            prediction = model.predict(X_new)[0]
            confidence = model.predict_proba(X_new)[0][int(prediction)] * 100

            st.subheader("Authentication Result:")
            if prediction == 0:
                st.success(f"This note appears to be GENUINE (Confidence: {confidence:.2f}%)")
            else:
                st.error(f"This note appears to be COUNTERFEIT (Confidence: {confidence:.2f}%)")

            st.subheader("Feature Analysis:")

            col1, col2 = st.columns(2)

            feature_names_list = list(features.keys())
            half = len(feature_names_list) // 2

            with col1:
                for feature_name in feature_names_list[:half]:
                    score = features[feature_name]
                    st.metric(
                        label=feature_name.replace('_', ' ').title(),
                        value=f"{score:.2f}",
                        delta="Good" if score > 0.8 else "Poor" if score < 0.5 else "Fair",
                        delta_color="normal" if 0.5 <= score <= 0.8 else "off"
                    )

            with col2:
                for feature_name in feature_names_list[half:]:
                    score = features[feature_name]
                    st.metric(
                        label=feature_name.replace('_', ' ').title(),
                        value=f"{score:.2f}",
                        delta="Good" if score > 0.8 else "Poor" if score < 0.5 else "Fair",
                        delta_color="normal" if 0.5 <= score <= 0.8 else "off"
                    )

            st.subheader("Visual Analysis:")

            vis_tabs = st.tabs([
                "Original", "Edges", "Segmentation", "Advanced Analysis"
            ])

            with vis_tabs[0]:
                st.image(input_image, caption="Original Image")

            with vis_tabs[1]:
                st.image(add_black_background(processed_images['edges']), caption="Edge Detection")

            with vis_tabs[2]:
                st.image(add_black_background(processed_images['threshold']), caption="Binary Segmentation")

            with vis_tabs[3]:
                col1, col2 = st.columns(2)
                with col1:
                    if processed_images['frequency'].size > 0:
                        st.image(add_black_background(processed_images['frequency']), caption="Frequency Domain Analysis")
                with col2:
                    if processed_images['latent'].size > 0:
                        st.image(add_black_background(processed_images['latent']), caption="Latent Image Analysis")

                if processed_images['histogram'].size > 0:
                    st.image(add_black_background(processed_images['histogram']), caption="Grayscale Histogram")

            st.subheader("Security Recommendations:")
            if prediction == 0:
                st.info("The currency note appears to be genuine. However, always verify currency using multiple methods when accepting high-value notes.")
            else:
                st.warning("This note appears to be counterfeit. We recommend:")
                st.markdown("""
                - Do not accept this note for transactions
                - Report to local authorities if you received this note
                - Check other notes from the same source
                - Verify using UV light and magnification for detailed assessment
                """)

    except Exception as e:
        # differentiate between empty or invalid images and other errors
        st.error(f"Error processing image: {str(e)}")
        st.info("Please upload a valid image file (JPEG/PNG) and try again.")

# Educational information
st.subheader("Educational Information")

with st.expander("How to verify Indian currency"):
    st.markdown("""
    ### Key security features of Indian currency notes:

    1. **Watermark**: Contains portrait of Mahatma Gandhi and the denomination numeral.

    2. **Security Thread**: Contains inscriptions 'भारत' and 'RBI' with color shift effect.

    3. **Latent Image**: The denomination numeral is visible when the note is held at 45° angle.

    4. **Micro-lettering**: Contains the word 'RBI' and the denomination value.

    5. **Intaglio Printing**: Raised printing on Mahatma Gandhi portrait, RBI seal, guarantee and promise text, Ashoka Pillar emblem.

    6. **Color-shifting Ink**: Denomination numeral changes color when viewed from different angles.

    7. **See-through Register**: Denomination numeral becomes complete when held against light.
    """)

with st.expander("Common counterfeit detection methods"):
    st.markdown("""
    ### Methods to detect counterfeit notes:

    1. **Feel test**: Genuine notes have a unique texture due to cotton-rag paper and intaglio printing.

    2. **Look test**: Check for clear watermarks, security thread, and see-through register when held against light.

    3. **Tilt test**: Check for color-shifting ink and latent image features.

    4. **UV light test**: Genuine notes show specific patterns under ultraviolet light.

    5. **Magnifying glass test**: Check for micro-lettering and fine details.
    """)

with st.expander("What to do if you suspect a counterfeit note"):
    st.markdown("""
    ### If you encounter a suspected counterfeit note:

    1. **Do not return it** to the person who gave it to you
    2. **Note the details** of the person who gave it to you
    3. **Handle it minimally** and place it in a protective covering
    4. **Report to the local police station** immediately
    5. **Contact your bank** for guidance

    Remember, knowingly passing counterfeit currency is a criminal offense under the Indian Penal Code sections 489A-489E.
    """)

# Footer
st.markdown("---")
st.markdown("© 2025 Indian Currency Authentication System. For educational purposes only.")
st.caption("This application is not an official tool of the Reserve Bank of India or any government agency.")