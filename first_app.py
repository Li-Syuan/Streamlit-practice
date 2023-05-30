import streamlit as st
import cv2
import numpy as np
from streamlit_cropper import st_cropper
from PIL import Image
st.set_page_config(layout='wide') # Set Streamlit to wide mode

def main():
    st.title('Image Thresholding and Adjustment')
    col1, col2, col3, col4 = st.columns(4)
    col1.subheader('Upload Image')
    col2.subheader('Operation')
    col3.subheader('Transform')
    col4.subheader('Morphology')
    
    with col1: 
        uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])
        _col1,_col2 = st.columns(2)
        with _col2:box_color = st.color_picker(label="Box Color", value='#0000FF')
        with _col1:aspect_choice = st.radio(label="Aspect Ratio", options=["1:1", "16:9", "4:3","Free"])
        aspect_dict = {"1:1": (1, 1),"16:9": (16, 9), "4:3": (4, 3),"Free": None}
        aspect_ratio = aspect_dict[aspect_choice]
    
    with col2:
        _col1, _col2 = st.columns(2)
        is_vf = _col1.checkbox('Vertical flip')
        is_hf = _col2.checkbox('Horizontal flip')
        is_cf = _col2.checkbox('Chanel flip')
        is_resize = _col1.checkbox('Resized')
        if is_resize: 
            _col1, _col2 = st.columns(2)
            w_size = _col1.number_input("Width",value = 512, min_value=64,step = 32)
            h_size = _col2.number_input("Height",value = 512, min_value=64,step = 32)

    with col2: 
        _col1, _col2 = st.columns(2)
        is_blur = _col1.checkbox('Blur')
        is_contours = _col2.checkbox('Draw contours')
        to_hsv = _col1.checkbox('Convert to HSV')
        is_convert2gray = _col1.checkbox('Convert to gray')
        bitwise_and = _col2.checkbox('Intersection')

    with col4:  
        kernel_size = st.slider('Kernel size', 0, 10, 0)
        kernel_size *= 2
        kernel_size -= 1
        kernel_size = max(1,kernel_size)
        kernel = np.ones([kernel_size,kernel_size])
        # col4, col5 = st.columns(2)
        erode_iter =  st.slider('Erode iteration', 0, 10, 0)
        dilate_iter =  st.slider('Dilate iteration', 0, 10, 0)

    with col3:
        # Brightness and contrast adjustment
        brightness_value = st.slider('Brightness', min_value=-255, max_value=255, value=0)
        contrast_value = st.slider('Contrast', min_value=-255, max_value=255, value=0)
        # Rotation
        angle = st.slider('Rotation Angle', min_value=-180, max_value=180, value=0)
        # Thresholding
        threshold_value = st.slider('Threshold Value', min_value=0, max_value=255, value=127)
   
    if uploaded_file is not None:
        # Read image
        img = Image.open(uploaded_file)
        _loc,_col,_ = st.columns([5,8,2])
        
        with _col:
            rect = st_cropper(
                img_file=img,
                box_color=box_color,
                aspect_ratio=aspect_ratio,
                return_type='box')
        left, top, width, height = tuple(map(int, rect.values()))
        img= np.asarray(img).astype('uint8')
        cropped_img = img[top:top + height, left:left + width]
        with _loc: st.write(rect)

        if is_convert2gray: cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)

        if len(cropped_img.shape) == 2:
            cropped_img = np.expand_dims(cropped_img, axis=-1)[:,:,[0] * 3]
        else:
            cropped_img = cropped_img[:,:,[2,1,0]]

        if is_resize: cropped_img = cv2.resize(cropped_img,(w_size,h_size))
        if is_vf: cropped_img = np.flip(cropped_img,axis = 0)
        if is_hf: cropped_img = np.flip(cropped_img,axis = 1)
        if is_cf: cropped_img = np.flip(cropped_img,axis = 2)

        # Convert to HSV
        if to_hsv: cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
        
        # Convert to grayscale
        gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        _, thresholded = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        thresholded = cv2.erode(thresholded,  kernel, iterations = erode_iter)
        thresholded = cv2.dilate(thresholded,  kernel, iterations = dilate_iter)
        
        # Draw the contours on the image
        adjusted = cv2.convertScaleAbs(cropped_img, alpha=1 + contrast_value/127.0, beta=brightness_value)
        height, width = cropped_img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
        adjusted = cv2.warpAffine(adjusted, rotation_matrix, (width, height))
        thresholded = cv2.warpAffine(thresholded, rotation_matrix, (width, height))

        # Apply Gaussian Blur
        if is_blur: adjusted = cv2.GaussianBlur(adjusted, (kernel_size, kernel_size), 0)
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
            st.image(cropped_img, channels='BGR')
        with col2:
            st.subheader('Adjusted Image')
            st.image(adjusted, channels='BGR')
        with col3:
            st.subheader('Thresholded Image')
            st.image(thresholded, channels='L')

if __name__ == '__main__':
    main()
