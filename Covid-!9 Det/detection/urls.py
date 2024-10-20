from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('about/', views.about, name='about'),
    path('upload/', views.upload_view, name='upload'),
    path('submit_success/<int:instance_id>/', views.submit_success, name='submit_success'),
    path('result/<int:instance_id>/', views.result_view, name='result'),
    path('images/', views.image_upload_list, name='image_upload_list'),
    path('register/', views.doctor_register, name='doctor_register'),
    path('login/', views.login, name='login'),
    path('logout/', views.logout_view, name='logout'),

]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
