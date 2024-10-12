import streamlit as st
import cv2
import numpy as np
from PIL import Image

from sklearn.preprocessing import StandardScaler

import joblib

def load_clustering_model():
    try:
        # Load the pre-trained model
        model = joblib.load('clustering_model.pkl')
        st.write("Model loaded successfully!")  # Log success
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure the model is saved as 'clustering_model.pkl'.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None


# Function to preprocess an uploaded image
def preprocess_image(image):
    image = np.array(image)  # Convert PIL image to NumPy array
    image = cv2.resize(image, (128, 128))

    # Proceed with grayscale conversion and other steps
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Histogram Equalization
    equalized_image = cv2.equalizeHist(gray_image)
    
    # Edge Detection (Canny)
    edges = cv2.Canny(equalized_image, 100, 200)
    
    # Gaussian Blurring
    blurred_image = cv2.GaussianBlur(edges, (5, 5), 0)

    # Normalize
    normalized_image = blurred_image / 255.0

    return normalized_image.astype(np.float32)


def extract_color_histogram(image):
    # Convert the image to uint8 if it's not already
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    # Ensure the image has 3 channels before calculating the histogram
    if len(image.shape) < 3 or image.shape[2] != 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert to 3 channels

    # Extract color histogram with 32 bins per channel
    hist = cv2.calcHist([image], [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()  # Normalize histogram

    return hist

from skimage.feature import graycomatrix, graycoprops

def extract_glcm_features(image):
    # Check if the image is already grayscale. If not, convert it.
    if len(image.shape) == 3 and image.shape[2] == 3:  # Check for 3-channel color image
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:  # Assume image is already grayscale or has a single channel
        gray_image = image
    
    gray_image = np.uint8(gray_image)  # Ensure image is in uint8 format
    glcm = graycomatrix(gray_image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    return [contrast, homogeneity, energy, correlation]

def extract_sift_features(image):
    # Ensure the image is in the correct format (uint8) before converting to grayscale
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    # Check if the image is already grayscale
    if len(image.shape) == 2:  # If image has only 2 dimensions (height, width), it's grayscale
        gray_image = image  # No need to convert
    else:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)
    if descriptors is not None:
        return descriptors.flatten()  # Mengembalikan deskriptor sebagai fitur
    else:
        return np.array([])  # Jika tidak ada deskriptor
    
def extract_combined_features(image):
    color_features = extract_color_histogram(image)
    glcm_features = extract_glcm_features(image)
    sift_features = extract_sift_features(image)

    # Gabungkan semua fitur
    combined_features = np.hstack([color_features, glcm_features, sift_features])
    return combined_features

def predict_cluster(image, model):
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Extract combined features
    features = extract_combined_features(processed_image)
    
    # Pad features to ensure consistent length
    max_len = max(len(features), 129028)  # Set max_len to either the length of the feature or the length expected by the model
    features_padded = np.pad(features, (0, max_len - len(features)), 'constant')  # Pad to max_len
    features = np.array([0 if val is None else val for val in features_padded])  # Ensure no None values

    # Normalize the features using StandardScaler
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform([features])  # Fit and transform features
    
    # Predict cluster
    cluster = model.predict([features])  # Assuming model has a `predict` method
    
    return cluster[0]

# Streamlit App
st.title("Image Clustering App")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])

# Load the clustering model
model = load_clustering_model()

if uploaded_file is not None and model is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file)
    
    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Perform clustering
    cluster = predict_cluster(image, model)
    
    # Display the cluster result
    st.write(f"The uploaded image belongs to Cluster: {cluster}")

    # Optionally, you can display relevant insights or visualizations for each cluster
    if cluster == 0:
        st.write("Cluster 0: This may represent empty land.")
    elif cluster == 1:
        st.write("Cluster 1: This may represent residential areas.")
    elif cluster == 2:
        st.write("Cluster 2: This may represent roads or industrial areas.")
else:
    if model is None:
        st.error("The model could not be loaded.")
