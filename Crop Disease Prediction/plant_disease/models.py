from django.db import models
from django.utils import timezone
from django.contrib.auth.models import User

# Create your models here.
from django.db import models

class PlantImage(models.Model):
    image = models.ImageField(upload_to='plant_images/')

class UserReg(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=100,default='Not Provided')
    created_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return self.user.username