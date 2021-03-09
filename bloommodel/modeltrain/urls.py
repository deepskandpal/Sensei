from django.urls import path

from . import views

app_name = 'modeltrain'
urlpatterns = [
    #path('', views.index, name='index'),
    path('', views.upload_csv, name='upload_csv'),
    path('train', views.train, name='train'),
]