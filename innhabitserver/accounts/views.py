from django.contrib.auth import get_user_model
from django.contrib.auth.decorators import permission_required
from django.http import HttpRequest, HttpResponse
from django.shortcuts import get_object_or_404, render

User = get_user_model()


@permission_required("accounts.view_user")
def users(request: HttpRequest) -> HttpResponse:
    users_objects = User.objects.all()
    context = {"users": users_objects}
    return render(request, "manage_users.html", context)


@permission_required("accounts.view_user")
def user(request: HttpRequest, id: int) -> HttpResponse:
    user_object = get_object_or_404(User, id=id)
    context = {"user": user_object}
    return render(request, "view_user.html", context)
