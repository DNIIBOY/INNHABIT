from django.db.models import Max
from django.http import HttpRequest, HttpResponse
from django.shortcuts import render
from occupancy.models import Entrance


def index(request: HttpRequest) -> HttpResponse:
    return render(request, "index.html")


def configuration(request: HttpRequest) -> HttpResponse:
    entrances = Entrance.objects.annotate(
        latest_entry=Max("entries__timestamp"),
        latest_exit=Max("exits__timestamp"),
    ).all()
    return render(request, "configuration.html", {"entrances": entrances})
