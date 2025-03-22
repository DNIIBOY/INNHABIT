from datetime import timedelta

from django.utils import timezone
from django_components import Component, register
from occupancy.models import EntryEvent


@register("daily_comparison")
class DailyComparison(Component):
    template_name = "daily_comparison.html"

    def get_context_data(self) -> dict:
        now = timezone.localtime()
        day_ago = now - timedelta(days=1)
        today_entries = EntryEvent.objects.filter(timestamp__date=now.date()).count()
        yesterday_entries = EntryEvent.objects.filter(
            timestamp__date=day_ago.date(),
            timestamp__lte=day_ago,  # Only count until current time yesterday
        ).count()

        diff = today_entries - yesterday_entries
        percentage = (diff / yesterday_entries) * 100 if yesterday_entries else 100

        return {
            "today_entries": today_entries,
            "yesterday_entries": yesterday_entries,
            "diff": diff,
            "percentage": int(abs(percentage)),
        }
