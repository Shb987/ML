import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from django.shortcuts import render, redirect
from django.core.files.storage import default_storage
from django.contrib import messages
from django.contrib.auth.models import User
from .models import UserReg
from django.utils import timezone
from django.contrib.auth import authenticate, login as auth_login, logout
from django.contrib.auth.decorators import login_required, user_passes_test
from django.http import JsonResponse


def home(request):
    return render(request, 'home.html')

def about(request):
    return render(request, 'about.html')


MODEL_PATH = os.path.join(os.path.dirname(__file__), 'plt.h5')

model = tf.keras.models.load_model(MODEL_PATH)

class_names = ['Pepper__bell___Bacterial_spot',
               'Pepper__bell___healthy',
               'Potato___Early_blight',
               'Potato___Late_blight',
               'Potato___healthy',
               'Tomato_Bacterial_spot',
               'Tomato_Early_blight',
               'Tomato_Late_blight',
               'Tomato_Leaf_Mold',
               'Tomato_Septoria_leaf_spot',
               'Tomato_Spider_mites_Two_spotted_spider_mite',
               'Tomato__Target_Spot',
               'Tomato__Tomato_YellowLeaf__Curl_Virus',
               'Tomato__Tomato_mosaic_virus',
               'Tomato_healthy']

class_names1 = {
    'Pepper__bell___Bacterial_spot': "Apply copper-based fungicides or bactericides. Remove and destroy infected plants to prevent spread.",
    'Pepper__bell___healthy': "No action required. Plant appears healthy.",
    'Potato___Early_blight': "Remove and destroy infected leaves. Apply fungicides if necessary.",
    'Potato___Late_blight': "Remove and destroy infected plants. Avoid overhead irrigation. Apply fungicides if necessary.",
    'Potato___healthy': "No action required. Plant appears healthy.",
    'Tomato_Bacterial_spot': "Remove and destroy infected leaves. Apply copper-based fungicides. Rotate crops to prevent recurrence.",
    'Tomato_Early_blight': "Remove and destroy infected leaves. Apply fungicides if necessary. Practice crop rotation.",
    'Tomato_Late_blight': "Remove and destroy infected plants. Apply fungicides. Avoid overhead irrigation.",
    'Tomato_Leaf_Mold': "Provide good air circulation. Avoid overhead irrigation. Apply fungicides.",
    'Tomato_Septoria_leaf_spot': "Remove and destroy infected leaves. Apply fungicides. Practice crop rotation.",
    'Tomato_Spider_mites_Two_spotted_spider_mite': "Apply miticides. Use insecticidal soap. Prune affected parts of the plant.",
    'Tomato__Target_Spot': "Remove and destroy infected leaves. Apply fungicides. Practice crop rotation.",
    'Tomato__Tomato_YellowLeaf__Curl_Virus': "Remove and destroy infected plants. Control whiteflies. Practice crop rotation.",
    'Tomato__Tomato_mosaic_virus': "Remove and destroy infected plants. Control aphids. Practice crop rotation.",
    'Tomato_healthy': "No action required. Plant appears healthy."
}

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(256, 256))  
    img_array = image.img_to_array(img)  
    img_array = np.expand_dims(img_array, axis=0)  
    return img_array

def predict_image(image_array):
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions[0])
    predicted_label = class_names[predicted_class]  
    return predicted_label

@login_required(login_url='login')
def upload_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']
        file_name = default_storage.save(image_file.name, image_file)
        file_path = default_storage.path(file_name)

        img_array = preprocess_image(file_path)
        predicted_label = predict_image(img_array)

        result = {
            'predicted_label': predicted_label,
            'remedy': class_names1.get(predicted_label, "No remedy available for this class.")
        }

        
        default_storage.delete(file_name)

        return JsonResponse(result)

    return render(request, 'upload.html')

def registration(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        email = request.POST.get('email')
        name = request.POST.get('name')
        password = request.POST.get('password')
        password1 = request.POST.get('password1')

     
        if password != password1:
            messages.error(request, "Your password and confirm password do not match!")
            return render(request, 'registration.html')

        try:
           
            if User.objects.filter(username=username).exists():
                messages.error(request, "Username already exists.")
                return render(request, 'registration.html')
            if User.objects.filter(email=email).exists():
                messages.error(request, "Email already exists.")
                return render(request, 'registration.html')

           
            user = User.objects.create_user(username=username, email=email, password=password)
            user.save()

            user_reg = UserReg.objects.create(user=user, name=name, created_at=timezone.now())
            user_reg.save()

            messages.success(request, "Registration successful! You can now log in.")
            return redirect('login')

        except Exception as e:
            messages.error(request, f'Something went wrong: {str(e)}')

    return render(request, 'registration.html')

def login_view(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')

        try:
            user = User.objects.get(email=email)
            user = authenticate(request, username=user.username, password=password)
            if user is not None:
                auth_login(request, user)
                next_url = request.GET.get('next', 'home')  
                return redirect(next_url)
            else:
                messages.error(request, "Invalid email or password.")
        except User.DoesNotExist:
            messages.error(request, "User with this email does not exist.")

    return render(request, 'login.html')



def logout_view(request):
    logout(request)
    return redirect('login')