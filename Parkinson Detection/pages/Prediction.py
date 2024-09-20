
import pickle
import streamlit as st
from streamlit_option_menu import option_menu 
import numpy as np
import os
import cv2
import pandas as pd
import pywt
from skimage.feature import hog, local_binary_pattern
from skimage.filters import gabor_kernel
from scipy import ndimage as ndi
from sklearn.model_selection import train_test_split
from skimage import io, color, exposure, transform, feature

# loading the saved models

spiral_model = pickle.load(open("rf_classifier.pkl",'rb'))
Wave_model = pickle.load(open("KNN_Wave.pickle",'rb'))
parkinsons_model = pickle.load(open("ml_rf_classifier.pkl",'rb'))

def preprocess_images(images):
        processed_images = []
        # Resize image to (200, 200)
        img_resized = cv2.resize(images, (200, 200))
        
        # # Thresholdinga
        thresholded_image = cv2.threshold(img_resized, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        
        processed_images.append(thresholded_image)
        return np.array(processed_images)




# Update the extract_features function to include DWT coefficients
def extract_features(images):
    features = []
    for img in images:
        # HOG features
        hog_features = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
        
        # LBP features
        lbp = local_binary_pattern(img, 8, 1, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9), density=True)
        
        # Gabor features
        frequency = 0.4
        thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        gabor_features = []
        for theta in thetas:
            gabor_real, gabor_imaginary = np.real(gabor_kernel(frequency, theta=theta)), np.imag(gabor_kernel(frequency, theta=theta))
            gabor_features.append(np.mean(ndi.convolve(img, gabor_real)))
            gabor_features.append(np.mean(ndi.convolve(img, gabor_imaginary)))
        
        # DWT coefficients
        coeffs = pywt.dwt2(img, 'haar')
        cA, (cH, cV, cD) = coeffs
        dwt_features = np.concatenate((cA.ravel(), cH.ravel(), cV.ravel(), cD.ravel()))
        
        # Concatenate all features
        all_features = np.concatenate((hog_features, lbp_hist, gabor_features, dwt_features))
        features.append(all_features)
    return np.array(features)

def quantify_image1(image_path):
    # Load the image
    image = io.imread(image_path)
    if image.shape[2] == 4:
        image = image[:, :, :3]  # Keep only the first three channels (RGB)
    # Convert the image to grayscale
    gray_image = color.rgb2gray(image)
    
    # Resize the image to a fixed size
    resized_image = transform.resize(gray_image, (200, 200))
    
    # Apply Histogram Equalization to enhance contrast
    equalized_image = exposure.equalize_hist(resized_image)
    
    # Compute HOG features for the preprocessed image
    hog_features = feature.hog(equalized_image, orientations=9,
                               pixels_per_cell=(10, 10),
                               cells_per_block=(2, 2),
                               transform_sqrt=True,
                               block_norm="L1")
    
    return hog_features

# sidebar navigation
with st.sidebar:
    
    selected = option_menu('Parkinson Disease Prediction System', 
                           ['Using Spiral Image  Prediction',
                            'Using Wave Pattern Image Prediction',
                            "Parkinson's Disease Prediction Using ML"],
                           icons=['image','image','person'],
                           default_index=0)
    
from PIL import Image
import cv2
# Heart Disease Prediction Page   
if (selected == 'Using Spiral Image  Prediction'): 
    # page title
    st.title('Parkinson Disease Prediction Using Spiral Image')
    
    uploaded_file=st.file_uploader('Choose an Image',type=['jpg','jpeg','png'])
    if uploaded_file  is not None:
        image=st.image(uploaded_file)
        img=Image.open(uploaded_file)
        img_gray = img.convert('L')
        image_shape = np.array(img).shape
        img_array = np.array(img_gray)
        image_preprocessed=preprocess_images(img_array)
        image_feutures=extract_features(image_preprocessed)
        image_feautures=image_feutures.reshape(1,-1)
        if st.button('Predict'):
            if image_shape != (256,256,3):
                st.error('Invalid Input')
            else:
                prediction=spiral_model.predict(image_feautures)
                if prediction== 0:
                    st.success("Result: The person does not have Parkinson's disease")
                else:
                    st.error("Result: The person have Parkinson's disease")
                







# Diabetes Prediction Page
if (selected == 'Using Wave Pattern Image Prediction'):
    
    st.title('Parkinson Disease Prediction Using Wave Pattern Image')
    
    uploaded_file=st.file_uploader('Choose an Image',type=['jpg','jpeg','png'])
    if uploaded_file  is not None:
        image=st.image(uploaded_file)
        img=Image.open(uploaded_file)
        image_shape = np.array(img).shape
        image_feutures=quantify_image1(uploaded_file)
       
        st.write(image_shape)
        # st.write(image_shape[0])
        # st.write(image_shape[1])
        # st.write(image_shape[2])
        if st.button('Predict'):
            if image_shape[1]<400 or image_shape[0]>400 :
                st.error('Invalid Input')
            else:
                prediction=Wave_model.predict([image_feutures])
                if prediction== 0:
                    st.success("Result: The person does not have Parkinson's disease")
                else:
                    st.error("Result: The person have Parkinson's disease")

  

    


# Parkinsons Prediction Page  
if (selected == "Parkinson's Disease Prediction Using ML"):    
    
    # page title
    st.title("Parkinson's Disease Prediction using ML")
    
    col1, col2, col3, col4, col5 = st.columns(5)  
    
    with col1:
        fo = st.text_input('MDVP: Fo(Hz)')
        
    with col2:
        fhi = st.text_input('MDVP: Fhi(Hz)')
        
    with col3:
        flo = st.text_input('MDVP: Flo(Hz)')
        
    with col4:
        Jitter_percent = st.text_input('MDVP: Jitter(%)')
        
    with col5:
        Jitter_Abs = st.text_input('MDVP: Jitter(Abs)')
        
    with col1:
        RAP = st.text_input('MDVP: RAP')
        
    with col2:
        PPQ = st.text_input('MDVP: PPQ')
        
    with col3:
        DDP = st.text_input('Jitter: DDP')
        
    with col4:
        Shimmer = st.text_input('MDVP: Shimmer')
        
    with col5:
        Shimmer_dB = st.text_input('MDVP: Shimmer(dB)')
        
    with col1:
        APQ3 = st.text_input('Shimmer: APQ3')
        
    with col2:
        APQ5 = st.text_input('Shimmer: APQ5')
        
    with col3:
        APQ = st.text_input('MDVP: APQ')
        
    with col4:
        DDA = st.text_input('Shimmer: DDA')
        
    with col5:
        NHR = st.text_input('NHR')
        
    with col1:
        HNR = st.text_input('HNR')
        
    with col2:
        RPDE = st.text_input('RPDE')
        
    with col3:
        DFA = st.text_input('DFA')
        
    with col4:
        spread1 = st.text_input('spread1')
        
    with col5:
        spread2 = st.text_input('spread2')
        
    with col1:
        D2 = st.text_input('D2')
        
    with col2:
        PPE = st.text_input('PPE')
        
    
    
    # code for Prediction
    parkinsons_diagnosis = ''
    
    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):
        if not all([fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]):
            st.warning("Please fill in all the fields.")
        else:
            parkinsons_prediction = parkinsons_model.predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ,DDP,Shimmer,Shimmer_dB,APQ3,APQ5,APQ,DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]])                          
            
            if (parkinsons_prediction[0] == 1):
              
              parkinsons_diagnosis = "The person has Parkinson's disease"
              st.error(parkinsons_diagnosis)
            else:
              parkinsons_diagnosis = "The person does not have Parkinson's disease"
              st.success(parkinsons_diagnosis)
        

    

    






    
    
    
    
    
    
    
