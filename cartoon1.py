import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to read and display the image
def read_file(uploaded_file):
    img = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)  # Decode image from the uploaded file
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    return img

# Function to create edge mask
def edge_mask(img, line_size, blur_value):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_blur = cv2.medianBlur(gray, blur_value)
    edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_size, blur_value)
    return edges

# Function for color quantization
def color_quantization(img, k):
    img = cv2.resize(img, (500, 300))  # Resize for faster processing
    data = np.float32(img).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
    ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(img.shape)
    return result

# Streamlit App
def main():
    st.title("Cartoonify Your Image")

    # Upload image file
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display original image
        img = read_file(uploaded_file)
        st.image(img, caption="Original Image", use_column_width=True)

        # Apply edge mask
        line_size, blur_value = 7, 7
        edges = edge_mask(img, line_size, blur_value)
        st.image(edges, caption="Edge Mask", use_column_width=True, clamp=True)

        # Apply color quantization
        img_quantized = color_quantization(img, k=9)
        st.image(img_quantized, caption="Quantized Image", use_column_width=True)

        # Apply bilateral filter to reduce noise
        blurred = cv2.bilateralFilter(img_quantized, d=3, sigmaColor=200, sigmaSpace=200)
        st.image(blurred, caption="Bilateral Filtered Image", use_column_width=True)

        # Combine edge mask with the blurred image to create the cartoon effect
        edges_resized = cv2.resize(edges, (blurred.shape[1], blurred.shape[0]))  # Resize edges to match blurred
        cartoon = cv2.bitwise_and(blurred, blurred, mask=edges_resized)

        # Display cartoonified image
        st.image(cartoon, caption="Cartoonified Image", use_column_width=True)

if __name__ == "__main__":
    main()
