from dashboard.components.dashboards.latest_events.latest_events import LatestEvents
from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="dashboard"),
    path("insights/", views.insights, name="insights"),
    path("components/latest_events/", LatestEvents.as_view(), name="latest_events"),
]
