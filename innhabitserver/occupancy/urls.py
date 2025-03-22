from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("configuration/", views.configuration, name="configuration"),
]
