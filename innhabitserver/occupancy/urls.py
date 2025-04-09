from dashboard.views import dates, edit_date, new_date
from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("export/", views.export_view, name="export"),
    path("export/csv/", views.export_data, name="export_csv"),
    path("configuration/", views.configuration, name="configuration"),
    path(
        "configuration/<int:pk>/", views.configure_entrance, name="configure_entrance"
    ),
    path("configuration/<int:device_id>/api_key/", views.api_key_view, name="api_key"),
    path("test_events/", views.test_events, name="test_events"),
    path("test_events/new/", views.select_test_entrance, name="select_test_entrance"),
    path("test_events/new/<int:pk>/", views.add_test_events, name="add_test_events"),
    path("dates/", dates, name="dates"),
    path("dates/new/", new_date, name="new_date"),
    path("dates/<int:pk>/", edit_date, name="edit_date"),
]
