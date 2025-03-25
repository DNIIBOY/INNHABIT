from dashboard.views import dates
from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("configuration/", views.configuration, name="configuration"),
    path(
        "configuration/<int:pk>/", views.configure_entrance, name="configure_entrance"
    ),
    path("configuration/<int:device_id>/api_key/", views.api_key_view, name="api_key"),
    path("dates/", dates, name="dates"),
]
