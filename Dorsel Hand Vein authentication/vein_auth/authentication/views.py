from django.shortcuts import render
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import pickle
import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pickle

import os
from PIL import Image
import cv2
from scipy.signal import convolve2d

from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.models import User
from .forms import RegistrationForm
from django.contrib.auth.decorators import login_required
# from PreProcessing_FeatureExtraction.normalize import normalize_data


dir = os.path.dirname(os.path.realpath('Untitled Folder'))
# print(dir)
filepath = os.path.join(dir, 'D:\\Projects\\dorsel_vscode\\PreProcessing_FeatureExtraction\\MexicanHatKernalData')
import numpy  as np


def normalize_data(x, low=0, high=1, data_type=None):
	x = np.asarray(x, dtype=np.float64)
	min_x, max_x = np.min(x), np.max(x)
	x = x - float(min_x)
	x = x / float((max_x - min_x))
	x = x * (high - low) + low
	if data_type is None:
		return np.asarray(x, dtype=np.float64)
	return np.asarray(x, dtype=data_type)



# print(filepath)
def remove_hair(image, kernel_size):
    # Ensure the input image is 2D (grayscale)
    if len(image.shape) == 3:  # If the image is RGB, convert to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Create a kernel
    kernel = np.ones((kernel_size, kernel_size), np.float32)
    normalized_kernel = kernel / kernel_size**2

    # Apply convolution
    hair_remove = convolve2d(image, normalized_kernel, mode='same', fillvalue=0)

    return hair_remove



dir = os.path.dirname(os.path.realpath('D:\\Projects\\dorsel_vscode\\PreProcessing_FeatureExtraction\\MexicanHatKernalData'))
print(dir)
filepath = os.path.join(dir, 'MexicanHatKernalData')



import numpy as np

import math


def compute_curvature(image, sigma):
	"""
	compute the curvature of profile in all 4 cross-section

	[Step 1-1] The objective of this step is, for all 4 cross-sections(z)
	of the image (horizontal, vertical, 45 and -45 ,diagonals) is computed then feed it to valley
	detector kappa function.

	kappa function

		kappa(z) = frac(d**2P_f(z)/dz**2,(1 + (dP_f(z)/dz)**2)**frac(3,2))

	To compute kappa function, first we smooth image using 2-dimensional gaussian
	filter to avoid noise from input dorsal data. We use Steerable Filters to smooth and get derivatives in
	higher order of smooth image, for all direction.

	Computing kappa vally detector function:
		1. construct a gaussian filter(h)
		2. take the first (dh/dx) and second (d^2/dh^2) derivatives of the filter
		3. calculate the first and second derivatives of the smoothed signal using
		derivative kernel's.
		:type image: object
		:param image, sigma:
		:return kappa:
	"""
	import scipy.ndimage as Image

	# 1. constructs the 2D gaussian filter "h" given the window size

	winsize = np.ceil(4 * sigma)  # enough space for the filter
	window = np.arange(-winsize, winsize + 1)
	X, Y = np.meshgrid(window, window)
	G = 1.0 / (2 * math.pi * sigma ** 2)
	G *= np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))

	# 2. calculates first and second derivatives of "G" with respect to "X"

	G1_0 = (-X / (sigma ** 2)) * G
	G2_0 = ((X ** 2 - sigma ** 2) / (sigma ** 4)) * G
	G1_90 = G1_0.T
	G2_90 = G2_0.T
	hxy = ((X * Y) / (sigma ** 8)) * G

	# 3. calculates derivatives w.r.t. to all directions
	image_g1_0 = 0.1 * Image.convolve(image, G1_0, mode='nearest')
	image_g2_0 = 10 * Image.convolve(image, G2_0, mode='nearest')
	image_g1_90 = 0.1 * Image.convolve(image, G1_90, mode='nearest')
	image_g2_90 = 10 * Image.convolve(image, G2_90, mode='nearest')
	fxy = Image.convolve(image, hxy, mode='nearest')
	image_g1_45 = 0.5 * np.sqrt(2) * (image_g1_0 + image_g1_90)
	image_g1_m45 = 0.5 * np.sqrt(2) * (image_g1_0 - image_g1_90)
	image_g2_45 = 0.5 * image_g2_0 + fxy + 0.5 * image_g2_90
	image_g2_m45 = 0.5 * image_g2_0 - fxy + 0.5 * image_g2_90

	# [Step 1-1] Calculation of curvature profiles

	#Hand_mask = mask.astype('float64')

	return np.dstack([
		(image_g2_0 / ((1 + image_g1_0 ** 2) ** (1.5))),
		(image_g2_90 / ((1 + image_g1_90 ** 2) ** (1.5))),
		(image_g2_45 / ((1 + image_g1_45 ** 2) ** (1.5))),
		(image_g2_m45 / ((1 + image_g1_m45 ** 2) ** (1.5))),
	])




def binaries(G):
	"""
	After connecting the veins center, we get a vein pattern in each direction.
	Then it is binaries using a median filter, for each corresponding pixel location,
	we calculate the median value. If the corresponding pixel is smaller than the calculated
	median, then it is the part of the background, and if the value of the pixel is larger or
	equal to the calculated median than it is the part of the vein pattern.
	Finally, we merge all four direction patterns into one by the corresponding pixel is
	replaced by the calculated median value at vein location.numpy
	:param G:
	:return:
	"""
	# take 1-D array and return bool mask based on its median value.
	median = np.median(G[G > 0])
	Gbool = G > median
	return Gbool.astype(np.float64)





def connect_profile_1d(vein_prob_1d):
	"""
	connect a 1d profile probabilistic score
	:param vein_prob_1d:
	:return: connected center
	"""

	return np.amin([np.amax([vein_prob_1d[3:-1], vein_prob_1d[4:]], axis=0),
	                   np.amax([vein_prob_1d[1:-3], vein_prob_1d[:-4]], axis=0)], axis=0)


def connect_centres(vein_score):
	""" 
	Connects vein centres by filtering vein probabilities V

	To connect the center position, to get a continues vein pattern,
	and to remove the noisy location of veins, we perform the following step-

	1.let's consider the horizontal direction, at any center location, say pixel(x, y),
	2. We consider two neighbor pixels, one in the right-hand side and another pixel on
	the left-hand side.
	3. If the value at both neighborhood right and left of the pixel(x,y) is large than
	the pixel value, then a horizontal line is drawn to form a continuous vein pattern,
	and if the values at both neighborhoods is less than the pixel(x,y),
	it is than considered as a noise, in this case, pixel (x,y) value is set to zero.
	4.Similarly, we calculate values in all directions to get continues patterns and remove noise.

	.. math::
		b[w] = min(max(a[w+1], a[w+2]) + max(a[w-1], a[w-2]))
	:type vein_score: object
	:param vein_score: all direction vein score
	:return connected_center: connected center in all direction
	"""
	#print("vein_score: ", vein_score.shape)
	connected_center = np.zeros(vein_score.shape, dtype='float64')
	temp = np.zeros((400, 400), dtype=np.float64)
	temp = vein_score[..., 0] + vein_score[..., 1] + vein_score[..., 2] + vein_score[..., 3]
	vein_score = temp
	# Horizontal direction
	for index in range(vein_score.shape[0]):
		connected_center[index, 2:-2, 0] = connect_profile_1d(vein_score[index, :])

	# Vertical direction
	for index in range(vein_score.shape[1]):
		connected_center[2:-2, index, 1] = connect_profile_1d(vein_score[:, index])

	#print(vein_score.shape)
	# Direction: 45 degrees (\)
	i, j = np.indices(vein_score.shape)
	border = np.zeros((2,), dtype='float64')
	for index in range(-vein_score.shape[0] + 5, vein_score.shape[1] - 4):
		connected_center[:, :, 2][i == (j - index)] = np.hstack(
			[border, connect_profile_1d(vein_score.diagonal(index)), border])

	# Direction: -45 degrees (/)
	Vud = np.flipud(vein_score)
	Cdud = np.flipud(connected_center[:, :, 3])
	for index in reversed(range(vein_score.shape[1] - 5, -vein_score.shape[0] + 4, -1)):
		Cdud[:, :][i == (j - index)] = np.hstack([border, connect_profile_1d(Vud.diagonal(index)), border])

	return connected_center





def profile_score_1d(profile_1d):
	"""
	1. we create a binary array by threshold the original array.
	2. We create a new array, which is moved by a pixel in the right
		direction of the threshold array.
	3. Subtract the new array from the threshold array and get another new array.
	4. When the value of the subtracted array is positive, it means that it is the beginning of curvature,
		and when the value of the array is negative it means that it is the end of curvature,
		we store all the starting and ending pairs. And the width of the curvature is measured
		by the length of the start end pair, and the depth of curvature is measured by the
		maximum value present at the location of the start end pairs in the original array.
		Finally, the location of the center is set by the midpoint of width.

	:param profile_1d
	:return: score_1d
	"""

	threshold_1d = (profile_1d > 0).astype(int)  # calculating mask where profile_1d > 0.
	diff = threshold_1d[1:] - threshold_1d[:-1]  # compute 1-shifted difference
	starts = np.argwhere(diff > 0)
	starts += 1  # compensates for shifted different
	ends = np.argwhere(diff < 0)
	ends += 1  # compensates for shifted different
	if threshold_1d[0]:
		starts = np.insert(starts, 0, 0)
	if threshold_1d[-1]:
		ends = np.append(ends, len(profile_1d))

	score_1d = np.zeros_like(profile_1d)

	if starts.size == 0 and ends.size == 0:
		return score_1d
	# computing and assigning probabilistic score.
	for start, end in zip(starts, ends):
		maximum = np.argmax(profile_1d[int(start):int(end)])
		score_1d[start + maximum] = profile_1d[start + maximum] * (end - start)
	return score_1d




def compute_vein_score(k):
	"""
	Evaluates joint vein centre probabilities from cross-sections

	This function take kappa and calculate vein centre probabilistic score
	based on whether kappa is positive or not. function work as follow:
	it consider each dimension of kappa(horizontal, vertical, diagonal etc.)
	then detect the centres of the veins and then based on width and maximum
	value in depth assign a probabilistic score.

	[Step 1-2] Detection of the centres of veins
	[Step 1-3] Assignment of scores to the centre positions
	[Step 1-4] Calculation of all the profiles

	:type k: object
	:param k: kappa function value for all direction
	:return score: probabilistic score for all direction
	"""

	# we have to return this variable correctly.
	score = np.zeros(k.shape, dtype='float64')
	# print(score.shape)
	# Horizontal direction
	for index in range(k.shape[0]):
		score[index, :, 0] += profile_score_1d(k[index, :, 0])

	# Vertical direction
	for index in range(k.shape[1]):
		score[:, index, 1] += profile_score_1d(k[:, index, 1])

	# Direction: 45 degrees (\)
	curve = k[:, :, 2]
	i, j = np.indices(curve.shape)  # taking indices of mesh.
	for index in range(-curve.shape[0] + 1, curve.shape[1]):
		score[i == (j - index), 2] += profile_score_1d(curve.diagonal(index))  # assigning value to diagonal.

	# Directionnp: -45 degrees (/)
	curve = np.flipud(k[:, :, 3])  # required so we get "/" diagonals correctly
	Vud = np.flipud(score)  # match above inversion
	for index in reversed(range(curve.shape[1] - 1, -curve.shape[0], -1)):
		Vud[i == (j - index), 3] += profile_score_1d(curve.diagonal(index))
	# print("Vud shape", Vud.shape)
	return score








def vein_pattern(image, kernel_size, sigma):

	"""
	In this method, the local maximum curvature is calculated in the cross-sectional
	profile of all four directions, then selecting the profile that has the maximum
	depth in the cross-sectional profile. And then to get the full pattern of nerves
	we add, The result of four directions.

	Miura et al. Proposed a three-step algorithm to solve the above problem.

	Step in Algorithms:

	Extraction of the center positions of veins.
	Connection of the center positions.
	Labeling of the image.

	:param image:
	:param kernel_size:
	:param sigma:
	:return: vein_pattern
	"""
	# data conversion to np.float64.
	data = np.asarray(image, dtype=np.float64)
	print("data shape", np.shape(data))
	# data preprocessing with remove hair.
	filter_data = remove_hair(data, kernel_size)

	# converting data to zero mean normalize form.
	preprocessed_data = normalize_data(filter_data, 0, 255)

	# detecting the rough location of vein. it reduce time complexity.
	#vein_mask = LeeMask(preprocessed_data)

	# STEP 1-1st: checking profile is dent or not using kappa value.
	kappa = compute_curvature(preprocessed_data, sigma=sigma)

	# STEP 1-2, 1-3, 1-4: assigning probabilistic score based on kappa values.
	score = compute_vein_score(kappa)

	# STEP 2nd: Connecting the center based on score.
	conect_score = connect_centres(score)

	# STEP 3rd: thresholding pattenrn based on median value.
	threshold = binaries(np.amax(conect_score, axis=2))

	# multiplying original data to binarise data to produce thick vein.
	vein_pattern = np.multiply(image, threshold, dtype=np.float64)

	return vein_pattern


import matplotlib.pyplot as plt


# image_path = 'D:\\Projects\\dorsel_vscode\\sample dataset\\Input\\s4\\2017228_2.jpg'
# image = cv2.imread(image_path, 0)
# processed_image = vein_pattern(image, 6, 8)

# plt.subplot(1,2,1)
# plt.imshow(image, cmap='gray')
# plt.title('Original Image')

# plt.subplot(1,2,2)
# plt.imshow(processed_image, cmap='gray')
# plt.title('Processed Image')

# plt.imshow(processed_image, cmap='gray')
# plt.axis('off')  # Turn off axis
# plt.savefig("S3_vein_pattern_extracted.png", bbox_inches='tight', pad_inches=0)  # Save the processed image without padding
# plt.show()

# plt.suptitle("Vein Pattern")
# plt.tight_layout()
# plt.savefig("vein_pattern_extracted_final.png")
# plt.show()


# plt.imshow(processed_image, cmap='gray')
# plt.axis('off')  # Turn off axis
# plt.savefig("S3_vein_pattern_extracted.png", bbox_inches='tight', pad_inches=0)  # Save the processed image without padding
# plt.show()





#£££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££

import tensorflow as tf
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import pickle


BATCH_SIZE = 16
IMAGE_SIZE = 224
CHANNELS = 3
EPOCHS = 10

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "veinpattern",
    seed=123,
    shuffle=True,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE
)

class_names = dataset.class_names

# Create a label encoder
label_encoder = LabelEncoder()

# Fit label encoder with class names
label_encoder.fit(class_names)

# Load the SVM classifier
@st.cache_resource
def load_svm_model():
    with open('Svm_model.pkl', 'rb') as f:
        loaded_model_svm = pickle.load(f)
    return loaded_model_svm

# Function to load the CNN model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('cnn_model.h5')  # Replace 'your_model.h5' with your model file

# Load models
model = load_model()
loaded_model_svm = load_svm_model()

# Set image dimensions
img_height, img_width = 224, 224

# Title and introduction
st.title('Dorsal Hand Vein-Based Authentication System')

st.write('Upload an image of your dorsal hand vein for authentication.')

# Upload image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Checkbox to enable/disable vein pattern extraction
    enable_extraction = st.checkbox("Enable Vein Pattern Extraction")

    # Perform authentication
    if st.button('Authenticate'):
        # Convert image to array and preprocess
        img = tf.keras.utils.load_img(
            uploaded_image, target_size=(img_height, img_width)
        )

        img_array = tf.keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, 0)  # Convert to numpy array and expand dimensions
        print(img_array.ndim, 'll')
        print(img_array.shape, 'll')
        # Apply vein pattern extraction if enabled
        if enable_extraction:
            resized_img = cv2.resize(img_array[0], (224, 224))
            gray_img = cv2.cvtColor(resized_img, cv2.COLOR_RGB2GRAY)
            print(gray_img.ndim, 'lk')
            img_array = vein_pattern(gray_img, 6, 8)
            print(img_array.shape, 'lk')
            img_array_batch = np.expand_dims(img_array, axis=0)
            img_array_batch_with_channels = np.expand_dims(img_array_batch, axis=-1)
            img_array_batch_with_channels = np.repeat(img_array_batch_with_channels, 3, axis=-1)
            print(img_array_batch_with_channels.shape,'l')
            img_array=img_array_batch_with_channels


        # Make predictions using CNN model
        predictions = model.predict(img_array)

        # Use the loaded SVM classifier to make predictions
        svm_probabilities = loaded_model_svm.predict_proba(predictions)

        # Get the predicted class index
        predicted_class_index = np.argmax(svm_probabilities)

        # Map the predicted index back to the original label
        predicted_class_label = label_encoder.classes_[predicted_class_index]
        predicted_class_probability = svm_probabilities[0, predicted_class_index]

        # Determine authentication result
        if predicted_class_probability > 0.05:
            st.success('Authentication successful! Access granted.')
            st.write(f"Predicted Class: {predicted_class_label}")
        else:
            st.error('Authentication failed! Access denied.')

@login_required(login_url='login')
def upload_and_authenticate(request):
    if request.method == 'POST' and request.FILES['image']:
        image = request.FILES['image']
        file_name = default_storage.save(image.name, ContentFile(image.read()))
        image_path = default_storage.path(file_name)

        
        img = Image.open(image_path)
        
        
        img = img.resize((224, 224))  
        img_array = np.array(img)

        if len(img_array.shape) == 2:  
            img_array = np.stack((img_array,)*3, axis=-1)
        
   
        img_array = np.expand_dims(img_array, axis=0)  

        if 'enable_extraction' in request.POST:
            resized_img = cv2.resize(img_array[0], (224, 224))
            gray_img = cv2.cvtColor(resized_img, cv2.COLOR_RGB2GRAY)
            img_array = vein_pattern(gray_img, 6, 8)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = np.expand_dims(img_array, axis=-1)
            img_array = np.repeat(img_array, 3, axis=-1) 

    
        model = load_model()
        loaded_model_svm = load_svm_model()

        predictions = model.predict(img_array)
        svm_probabilities = loaded_model_svm.predict_proba(predictions)
        predicted_class_index = np.argmax(svm_probabilities)
        predicted_class_label = label_encoder.classes_[predicted_class_index]
        predicted_class_probability = svm_probabilities[0, predicted_class_index]


        if predicted_class_probability > 0.05:
            return render(request, 'result.html', {'result': 'Success','label': predicted_class_label})
        else:
            return render(request, 'result.html', {'result': 'Failed'})
    return render(request, 'upload.html')

def home_page(request):
	return render(request,'home.html')

def about_view(request):
	return render(request,'about.html')


def register_view(request):
    if request.method == 'POST':
        form = RegistrationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('login')
    else:
        form = RegistrationForm()
    return render(request, 'register.html', {'form': form})

def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('home')  
    else:
        form = AuthenticationForm()
    return render(request, 'login.html', {'form': form})

def logout_view(request):
    logout(request)
    return redirect('login')
