from datetime import timedelta
from typing import Any

from django.http import HttpRequest, HttpResponse
from django.template.context import Context
from django.utils import timezone
from django_components import Component, register
from occupancy.models import EntryEvent


@register("daily_visitors")
class DailyVisitors(Component):
    template_name = "daily_visitors.html"

    def get(self, request: HttpRequest) -> HttpResponse:
        return self.render_to_response(request=request)

    def get_template_data(
        self, args: Any, kwargs: Any, slots: Any, context: Context
    ) -> dict:
        weekdays = ["Man", "Tir", "Ons", "Tor", "Fre", "Lør", "Søn"]

        now = timezone.localtime()
        yesterday = now - timedelta(days=1)

        entries = EntryEvent.objects.filter(
            timestamp__date=now.date(),
            timestamp__lte=now,
        ).count()

        yesterday_entries = EntryEvent.objects.filter(
            timestamp__date=yesterday.date(),
            timestamp__lte=yesterday,
        ).count()

        if yesterday_entries == 0:
            yesterday_entries = 1  # Avoid division by zero

        return {
            "daily_visitors": entries,
            "yesterday_visitors": yesterday_entries,
            "yesterday_diff": entries - yesterday_entries,
            "yesterday_percentage": round(
                (entries - yesterday_entries) / yesterday_entries * 100, 2
            ),
            "date": now.date(),
            "weekday": weekdays[now.date().weekday()],
        }
