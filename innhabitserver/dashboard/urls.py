from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="dashboard-index"),
    path("insights/", views.insights, name="insights-dashboard"),
]
