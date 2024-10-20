from django.shortcuts import render, get_object_or_404, redirect
from .forms import ImageUploadForm
from .models import ImageUpload
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np
import os
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib.auth.hashers import make_password
from django.contrib.auth import authenticate, login as auth_login, logout
from .models import DoctorReg

def load_model():
    return tf.keras.models.load_model(r'C:\Users\shiha\OneDrive\Desktop\covid_detection22\covid_detection\Covid_Detection.h5')

model = load_model()

@login_required
def upload_view(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        age = request.POST.get('age')
        sex = request.POST.get('sex')
        image_file = request.FILES['image']

        img_instance = ImageUpload(name=name, age=age, sex=sex, image=image_file, doctor=request.user.username)
        img_instance.save()
        img_path = os.path.join(settings.MEDIA_ROOT, img_instance.image.name)

        img = keras_image.load_img(img_path, target_size=(224, 224))
        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)
        result = 'COVID' if prediction[0][0] > prediction[0][1] else 'Non-COVID'
        
        img_instance.result = result
        img_instance.save()

        return redirect('submit_success', instance_id=img_instance.id)
    return render(request, 'detection/upload.html')

def submit_success(request, instance_id):
    instance = ImageUpload.objects.get(id=instance_id)
    return render(request, 'detection/submit_success.html', {'instance': instance})

def submit_success(request, instance_id):
    instance = ImageUpload.objects.get(id=instance_id)
    return render(request, 'detection/submit_success.html', {'instance': instance})

def result_view(request, instance_id):
    instance = ImageUpload.objects.get(id=instance_id)
    return render(request, 'detection/result.html', {'instance': instance})

@login_required
def image_upload_list(request):
    images = ImageUpload.objects.all()
    return render(request, 'detection/image_upload_list.html', {'images': images})

def doctor_register(request):
    if request.method == 'POST':
        username = request.POST['username']
        email = request.POST['email']
        name = request.POST['name']
        qualification = request.POST['qualification']
        password = request.POST['password']
        confirm_password = request.POST['confirm_password']
        
        errors = []
        
        if password != confirm_password:
            errors.append("Passwords do not match.")
        
        if User.objects.filter(username=username).exists():
            errors.append("Username already taken.")
        
        if User.objects.filter(email=email).exists():
            errors.append("Email already taken.")
        
        if errors:
            return render(request, 'detection/doctor_reg.html', {'errors': errors})
        
        user = User.objects.create(
            username=username,
            email=email,
            first_name=name,
            password=make_password(password)
        )
        
        DoctorReg.objects.create(
            user=user,
            name=name,
            qual=qualification  
        )
        
        return redirect('login')  

    return render(request, 'detection/doctor_reg.html')

def login(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        
        user = authenticate(request, username=username, password=password)
        if user is not None:
            auth_login(request, user)
            return redirect('image_upload_list')  
        else:
            return render(request, 'detection/login.html', {'error': 'Invalid username or password.'})

    return render(request, 'detection/login.html')

def home(request):
    return render(request, 'detection/home.html')

def about(request):
    return render(request, 'detection/about.html')

def logout_view(request):
    logout(request)
    return redirect('login')
