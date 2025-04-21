import random

from dashboard.models import LabelledDate
from django.contrib.auth.decorators import login_required, permission_required
from django.core.exceptions import ValidationError
from django.http import HttpRequest, HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views.decorators.http import require_http_methods
from django_components import DynamicComponent
from occupancy.forms import LabelledDateForm


def index(request: HttpRequest) -> HttpResponse:
    if request.user.is_authenticated:
        return render(request, "admin_dashboard.html")
    return render(request, "public_dashboard.html")


@login_required
def top_row(request: HttpRequest) -> HttpResponse:
    return render(request, "admin_dashboard.html#top_row")


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

    index_arg = request.GET.get("index", None)
    if index_arg:
        insight = components[int(index_arg) % len(components)]
        context = {"insight_name": insight, "dashboard": True}
        return render(request, "insights.html", context=context)

    insight = request.GET.get("insight", None)
    if insight:
        locked_insight = True
        if insight not in components:
            raise ValidationError("Invalid insight")
    else:
        locked_insight = False
        insight = random.choice(components)

    if request.htmx:
        return DynamicComponent.render_to_response(
            kwargs={"is": insight},
            request=request,
        )
    context = {"insight_name": insight, "locked_insight": locked_insight}
    return render(request, "insights.html", context=context)


@permission_required("dashboard.view_labelleddate")
def dates(request: HttpRequest) -> HttpResponse:
    date_objects = LabelledDate.objects.order_by("date").all()
    return render(request, "dates.html", {"dates": date_objects})


@permission_required("dashboard.add_labelleddate")
@require_http_methods(["GET", "POST"])
def new_date(request: HttpRequest) -> HttpResponse:
    if request.method == "GET":
        return render(request, "new_date.html")
    form = LabelledDateForm(request.POST)
    if not form.is_valid():
        return render(request, "new_date.html", {"form": form})
    LabelledDate.objects.create(**form.cleaned_data)
    return redirect("dates")


@require_http_methods(["GET", "POST"])
def edit_date(request: HttpRequest, pk: int) -> HttpResponse:
    labelled_date = get_object_or_404(LabelledDate, pk=pk)
    if request.method == "GET":
        return render(request, "new_date.html", {"date": labelled_date})
    form = LabelledDateForm(request.POST)
    if not form.is_valid():
        return render(request, "new_date.html", {"form": form, "date": labelled_date})
    labelled_date.date = form.cleaned_data["date"]
    labelled_date.label = form.cleaned_data["label"]
    labelled_date.save()
    return redirect("dates")
