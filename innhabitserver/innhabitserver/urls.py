from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("__reload__/", include("django_browser_reload.urls")),
    path("", include("occupancy.urls")),
    path("accounts/", include("django.contrib.auth.urls")),
    path("admin/", admin.site.urls),
    path("api/v1/", include("api.urls")),
    path("dashboard/", include("dashboard.urls")),
]
