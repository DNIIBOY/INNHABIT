from datetime import timedelta

from django.utils import timezone
from django_components import Component, register
from occupancy.models import EntryEvent


@register("weekly_visitors")
class WeeklyVisitors(Component):
    template_name = "weekly_visitors.html"

    def get_context_data(self) -> dict:
        today = timezone.localtime().date()
        start_of_this_week = today - timedelta(days=today.weekday())
        end_of_this_week = start_of_this_week + timedelta(days=6)
        start_of_last_week = start_of_this_week - timedelta(days=7)
        end_of_last_week = end_of_this_week - timedelta(days=7)

        entries = EntryEvent.objects.filter(
            timestamp__date__range=(start_of_this_week, end_of_this_week)
        ).count()

        last_week_entries = EntryEvent.objects.filter(
            timestamp__date__range=(start_of_last_week, end_of_last_week)
        ).count()

        if last_week_entries == 0:
            last_week_entries = 1  # Avoid division by zero

        return {
            "weekly_visitors": entries,
            "last_week_visitors": last_week_entries,
            "last_week_diff": entries - last_week_entries,
            "last_week_percentage": round(
                (entries - last_week_entries) / last_week_entries * 100, 2
            ),
            "date": today,
            "week_number": today.isocalendar()[1],
        }
