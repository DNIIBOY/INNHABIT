from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("configuration/", views.configuration, name="configuration"),
    path("configuration/<int:pk>", views.configure_entrance, name="configure_entrance"),
]
