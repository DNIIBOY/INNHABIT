from datetime import timedelta

from django.http import HttpRequest, HttpResponse
from django.utils import timezone
from django_components import Component, register
from occupancy.models import EntryEvent


@register("daily_visitors")
class DailyVisitors(Component):
    template_name = "daily_visitors.html"

    def get(self, request: HttpRequest) -> HttpResponse:
        return self.render_to_response(request=request)

    def get_context_data(self) -> dict:

        weekdays = ["Man", "Tir", "Ons", "Tor", "Fre", "Lør", "Søn"]

        today = timezone.localtime().date()
        start_of_this_day = today - timedelta(days=today.weekday())
        end_of_this_day = start_of_this_day + timedelta(days=6)
        start_of_last_day = start_of_this_day - timedelta(days=7)
        end_of_last_day = end_of_this_day - timedelta(days=7)

        entries = EntryEvent.objects.filter(
            timestamp__date__range=(start_of_this_day, end_of_this_day)
        ).count()

        last_day_entries = EntryEvent.objects.filter(
            timestamp__date__range=(start_of_last_day, end_of_last_day)
        ).count()

        if last_day_entries == 0:
            last_day_entries = 1  # Avoid division by zero

        return {
            "daily_visitors": entries,
            "last_day_visitors": last_day_entries,
            "last_day_diff": entries - last_day_entries,
            "last_day_percentage": round(
                (entries - last_day_entries) / last_day_entries * 100, 2
            ),
            "date": today,
            "weekday": weekdays[today.weekday()],
        }
