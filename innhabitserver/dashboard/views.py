from django.shortcuts import render
from django.http import HttpRequest, HttpResponse


def index(request: HttpRequest) -> HttpResponse:
    if request.user.has_perm("accounts.view_admin_dashboard"):
        return render(request, "admin_dashboard.html")
    return render(request, "public_dashboard.html")


def insights(request: HttpRequest) -> HttpResponse:
    return render(request, "insights.html")
