from django.urls import path
from .views import upload_image,home,about,registration,login_view,logout_view

urlpatterns = [
    path('', home, name='home'),
    path('upload/', upload_image, name='upload_image'),
    path('about/', about, name='about'), 
    path('register/', registration, name='register'),
    path('login/', login_view, name='login'),
    path('logout/', logout_view, name='logout'),

]
