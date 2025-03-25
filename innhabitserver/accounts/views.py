from django.contrib.admin.models import ADDITION, CHANGE
from django.contrib.auth import get_user_model
from django.contrib.auth.decorators import login_required, permission_required
from django.core.exceptions import PermissionDenied
from django.http import HttpRequest, HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils.encoding import force_str
from django.utils.http import urlsafe_base64_decode

from .forms import AddUserForm, SetupUserForm
from .tokens import account_activation_token
from .utils import log_admin_action, send_activation_email

User = get_user_model()


@permission_required("accounts.view_user")
def users(request: HttpRequest) -> HttpResponse:
    users_objects = User.objects.all()
    context = {
        "users": users_objects,
    }
    return render(request, "manage_users.html", context)


@permission_required("accounts.add_user")
def add_user(request: HttpRequest) -> HttpResponse:
    if request.method == "GET":
        return render(request, "registration/add_user.html")

    form = AddUserForm(request.POST)
    if not form.is_valid():
        return render(request, "registration/add_user.html", {"form": form})

    user_object = User.objects.create_user(
        email=form.cleaned_data["email"],
        is_active=False,
    )
    send_activation_email(user_object, host=request.get_host())
    log_admin_action(request.user, user_object, ADDITION, [{"added": {}}])
    return redirect("users")


@permission_required("accounts.view_user")
def user(request: HttpRequest, user_id: int) -> HttpResponse:
    print(request.method)
    user_object = get_object_or_404(User, id=user_id)
    if request.method == "DELETE":
        if not request.user.has_perm("accounts.delete_user"):
            raise PermissionDenied
        user_object.is_active = False
        user_object.save()
        log_admin_action(
            request.user, user_object, CHANGE, [{"changed": {"fields": ["is_active"]}}]
        )
    context = {"viewed_user": user_object}
    template = "view_user.html"
    if request.htmx:
        template += "#user_page"
    return render(request, template, context)


@login_required
def profile(request: HttpRequest) -> HttpResponse:
    user_object = request.user
    context = {"user": user_object}
    return render(request, "view_user.html", context)


def activate_account_view(
    request: HttpRequest, b64uid: str, token: str
) -> HttpResponse:
    if request.user.is_authenticated:
        return redirect("index")

    uid = force_str(urlsafe_base64_decode(b64uid))
    user_object = User.objects.filter(id=uid, is_active=False).first()
    if not user_object or not account_activation_token.check_token(user_object, token):
        return render(request, "registration/activation_failure.html")

    if request.method == "GET":
        context = {"new_user": user_object}
        return render(request, "registration/setup_account.html", context)

    form = SetupUserForm(request.POST, instance=user_object)
    if not form.is_valid():
        context = {"new_user": user_object, "form": form}
        return render(request, "registration/setup_account.html", context)

    form.save()
    user_object.is_active = True
    user_object.save()
    return redirect("login")
