from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone


class ImageUpload(models.Model):
    name = models.CharField(max_length=100, default="Not Provided")
    age = models.IntegerField(default=0)
    sex = models.CharField(max_length=10, default="Unknown")
    image = models.ImageField(upload_to='uploads/')
    result = models.CharField(max_length=100, blank=True, null=True)
    doctor = models.CharField(max_length=100, blank=True, null=True) 
    
    def __str__(self):
        return f"{self.name} - {self.image.name}"
    

class DoctorReg(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=100,default='Not Provided')
    qual = models.CharField(max_length=100,blank=True, null=True)

    created_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return self.user.username