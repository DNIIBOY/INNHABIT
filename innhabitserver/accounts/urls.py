from django.urls import path

from . import views

urlpatterns = [
    path("users/", views.users, name="users"),
    path("users/<int:user_id>/", views.user, name="user"),
    path("users/add/", views.add_user, name="add_user"),
    path(
        "activate/<str:b64uid>/<token>/",
        views.activate_account_view,
        name="activate_account",
    ),
]
