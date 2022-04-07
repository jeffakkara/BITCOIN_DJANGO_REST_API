from django.urls import path, include
from . import views
from rest_framework import routers


urlpatterns = [
    path('predict_single/', views.predict),
    path('predict_range/', views.predict_range),
    path('predict_from_sample/', views.predict_from_sample)
 
]