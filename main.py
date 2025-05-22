import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image

# --- Color Extraction Logic ---

def extract_colors(image, num_colors=5):
    reshaped = image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(reshaped)
    return kmeans.cluster_centers_

def show_color_bar(colors):
    bar_height = 50
    bar_width = 60 * len(colors)
    bar = np.zeros((bar_height, bar_width, 3), dtype='uint8')

    for i, color in enumerate(colors):
        start = i * 60
        end = (i + 1) * 60
        cv2.rectangle(bar, (start, 0), (end, bar_height), color.astype('uint8').tolist(), -1)
    
    return bar

# --- Streamlit UI ---

st.set_page_config(page_title="Color Detector", layout="centered")
st.title("🎨 Color Detection from Image")
st.write("Upload an image or take a photo to detect dominant colors.")

option = st.radio("Choose image input method:", ["Upload Image", "Take Photo"])

image = None

if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

elif option == "Take Photo":
    image_data = st.camera_input("Take a picture")
    if image_data:
        image = Image.open(image_data)
        st.image(image, caption="Captured Image", use_container_width=True)

# --- Process Image ---
if image:
    img_array = np.array(image)
    small_img = cv2.resize(img_array, (100, 100))
    colors = extract_colors(small_img)

    st.subheader("Detected Dominant Colors")
    color_bar = show_color_bar(colors)

    st.image(color_bar, caption="Dominant Colors", use_container_width=False)

    # Show hex values
    st.subheader("Hex Color Codes")
    hex_colors = ['#%02x%02x%02x' % tuple(map(int, color)) for color in colors]
    st.write(hex_colors)
