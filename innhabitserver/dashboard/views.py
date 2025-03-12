import random
from django.shortcuts import render
from django.http import HttpRequest, HttpResponse
from django.core.exceptions import ValidationError


def index(request: HttpRequest) -> HttpResponse:
    if request.user.has_perm("accounts.view_admin_dashboard"):
        return render(request, "admin_dashboard.html")
    return render(request, "public_dashboard.html")


def insights(request: HttpRequest) -> HttpResponse:
    components = [
        "current_occupancy",
        "cafeteria_day",
    ]
    insight = request.GET.get("insight", None)
    if insight:
        if insight not in components:
            raise ValidationError("Invalid insight")
    else:
        insight = random.choice(components)
    context = {"insight_name": insight}
    return render(request, "insights.html", context=context)
