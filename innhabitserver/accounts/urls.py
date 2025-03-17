from django.urls import path

from . import views

urlpatterns = [
    path("users/", views.users),
    path("users/<int:user_id>/", views.user),
    path(
        "activate/<str:b64uid>/<token>/",
        views.activate_account_view,
        name="activate_account",
    ),
]
