from django.urls import path
# from .views import upload_and_authenticate
from .import views

urlpatterns = [
    path('upload/', views.upload_and_authenticate, name='upload_and_authenticate'),
    path('',views.home_page,name='home'),
    path('home/',views.home_page,name='home'),
    path('about/',views.about_view,name='about'),
    path('register/', views.register_view, name='register'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    
]
