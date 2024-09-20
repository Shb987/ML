from django.contrib import admin

# Register your models here.
from .models import PlantImage,UserReg

admin.site.register(PlantImage) 
admin.site.register(UserReg) 