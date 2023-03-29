import streamlit as st
import cv2
import numpy as np
st.set_page_config(layout='wide') # Set Streamlit to wide mode
def main():
    st.title('Image Thresholding and Adjustment Demo')

    # Upload image
    st.subheader('Upload Image')
    col1, col2, col3 = st.columns([2,1,1])
    with col1: uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])
    with col2: 
        col4, col5 = st.columns(2)
        is_resize = col4.checkbox('Resized')
        is_blur = col5.checkbox('Blur')
        is_contours = col4.checkbox('Draw contours')
        is_cf = col5.checkbox('Chanel flip')
        is_vf = col4.checkbox('Vertical flip')
        is_hf = col5.checkbox('Horizontal flip')
        to_hsv = col4.checkbox('Convert to HSV')
        bitwise_and = col5.checkbox('Intersection')
    with col3: 
        if is_resize: 
            col4, col5 = st.columns(2)
            w_size = col4.number_input("Width",value = 512, min_value=64)
            h_size = col5.number_input("Height",value = 512, min_value=64)
        if is_blur:
            kernel_size = st.slider('Kernel size', 1, 25, 5)


    col1, col2, col3, col4 = st.columns(4)
    if uploaded_file is not None:
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if is_resize: img = cv2.resize(img,(w_size,h_size))
        if is_vf: img = np.flip(img,axis = 0)
        if is_hf: img = np.flip(img,axis = 1)
        if is_cf: img = np.flip(img,axis = 2)

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
        
        # Draw the contours on the image
        adjusted = cv2.convertScaleAbs(img, alpha=1 + contrast_value/127.0, beta=brightness_value)

        
        height, width = img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
        adjusted = cv2.warpAffine(adjusted, rotation_matrix, (width, height))
        thresholded = cv2.warpAffine(thresholded, rotation_matrix, (width, height))

        # Apply Gaussian Blur
        if is_blur:
            kernel_size *= 2
            kernel_size -= 1
            adjusted = cv2.GaussianBlur(adjusted, (kernel_size, kernel_size), 0)
        # Find the contours
        if is_contours:
            contours, _ = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(adjusted, contours, -1, (0, 255, 0), 3)
        
        if bitwise_and: 
            b,g,r = cv2.split(adjusted)
            b = cv2.bitwise_and(b, thresholded)
            g = cv2.bitwise_and(g, thresholded)
            r = cv2.bitwise_and(r, thresholded)
            adjusted  = cv2.merge((b,g,r))

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
