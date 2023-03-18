import streamlit as st
import cv2
import numpy as np
st.set_page_config(layout='wide') # Set Streamlit to wide mode
def main():
    st.title('Image Thresholding and Adjustment Demo')

    # Upload image
    st.subheader('Upload Image')
    col1, col2 = st.columns(2)
    with col1: uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])
    with col2: to_hsv = st.checkbox('Convert to HSV')
    col1, col2, col3, col4 = st.columns(4)
    if uploaded_file is not None:
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Brightness and contrast adjustment
        with col1: brightness_value = st.slider('Brightness', min_value=-255, max_value=255, value=0)
        with col2: contrast_value = st.slider('Contrast', min_value=-127, max_value=127, value=0)
         # Rotation
        with col3: angle = st.slider('Rotation Angle', min_value=-180, max_value=180, value=0)
        # Thresholding
        with col4: threshold_value = st.slider('Threshold Value', min_value=0, max_value=255, value=127)
        # Convert to HSV
        if to_hsv: img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresholded = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        
        adjusted = cv2.convertScaleAbs(img, alpha=1 + contrast_value/127.0, beta=brightness_value)
        
        height, width = img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
        adjusted = cv2.warpAffine(adjusted, rotation_matrix, (width, height))
   
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader('Original Image')
            st.image(img, channels='BGR')
        with col2:
            st.subheader('Adjusted Image')
            st.image(adjusted, channels='BGR')
        with col3:
            st.subheader('Thresholded Image')
            st.image(thresholded, channels='L')


if __name__ == '__main__':
    main()
