from django.db import models

# Create your models here.

class VeinImage(models.Model):
    user = models.ForeignKey('auth.User', on_delete=models.CASCADE)
    image = models.ImageField(upload_to='vein_images/')
    processed = models.BooleanField(default=False)
    result = models.CharField(max_length=50, null=True, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)

from django.contrib.auth.models import User
from django.db import models

class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)

