from typing import Callable
from zoneinfo._common import ZoneInfoNotFoundError

from django.http import HttpRequest, HttpResponse
from django.utils import timezone


class TimezoneMiddleware:
    def __init__(self, get_response: Callable) -> None:
        self.get_response = get_response

    def __call__(self, request: HttpRequest) -> HttpResponse:
        tzname = request.COOKIES.get("tzinfo")
        try:
            timezone.activate(tzname)
        except (ZoneInfoNotFoundError, ValueError):
            timezone.deactivate()

        return self.get_response(request)
