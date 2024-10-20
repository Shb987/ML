from django.contrib import admin

# Register your models here.
from .models import ImageUpload,DoctorReg

admin.site.register(ImageUpload) 
admin.site.register(DoctorReg)