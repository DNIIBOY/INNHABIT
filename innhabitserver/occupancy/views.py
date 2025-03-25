from django.db.models import F, Max
from django.db.models.functions import Greatest
from django.http import HttpRequest, HttpResponse
from django.shortcuts import get_object_or_404, render
from occupancy.models import Entrance


def index(request: HttpRequest) -> HttpResponse:
    return render(request, "index.html")


def configuration(request: HttpRequest) -> HttpResponse:
    entrances = (
        Entrance.objects.annotate(
            latest_entry=Max("entries__timestamp"),
            latest_exit=Max("exits__timestamp"),
            latest_event=Greatest(F("latest_entry"), F("latest_exit")),
        )
        .order_by("id")
        .all()
    )
    return render(request, "configuration.html", {"entrances": entrances})


def configure_entrance(request: HttpRequest, pk: int) -> HttpRequest:
    entrance = get_object_or_404(Entrance, pk=pk)
    context = {
        "entrance": entrance,
        "device": getattr(entrance, "device", None),
    }
    return render(request, "configure_entrance.html", context)
