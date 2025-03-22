import random

from dashboard.models import LabelledDate
from django.core.exceptions import ValidationError
from django.http import HttpRequest, HttpResponse
from django.shortcuts import render
from django.utils import timezone
from django_components import DynamicComponent


def index(request: HttpRequest) -> HttpResponse:
    if request.user.is_authenticated:
        return render(request, "admin_dashboard.html")
    return render(request, "public_dashboard.html")


def insights(request: HttpRequest) -> HttpResponse:
    today = timezone.localtime().date()
    components = [
        "cafeteria_day",
        "current_occupancy",
        "daily_comparison",
        "top_days",
        "visitors_today",
    ]
    if LabelledDate.objects.filter(date__gt=today).exists():
        components.append("upcoming_event")

    locked_insight = False
    insight = request.GET.get("insight", None)
    if insight:
        if insight not in components:
            raise ValidationError("Invalid insight")
        locked_insight = True
    else:
        insight = random.choice(components)

    if request.htmx:
        return DynamicComponent.render_to_response(
            kwargs={"is": insight},
            request=request,
        )
    context = {"insight_name": insight, "locked_insight": locked_insight}
    return render(request, "insights.html", context=context)
