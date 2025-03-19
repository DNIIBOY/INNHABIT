import datetime

from django.db.models import Count
from django.db.models.functions import ExtractWeekDay
from django.utils import timezone
from django_components import Component, register
from occupancy.models import EntryEvent


@register("cafeteria_day")
class CafeteriaDay(Component):
    template_name = "cafeteria_day.html"

    def get_context_data(self) -> dict:
        today = timezone.localtime().date()
        start_of_this_week = today - datetime.timedelta(days=today.weekday())
        end_of_prev_week = start_of_this_week - datetime.timedelta(days=1)
        start_of_prev_week = end_of_prev_week - datetime.timedelta(days=6)

        top_day = (
            EntryEvent.objects.filter(
                timestamp__date__range=(start_of_prev_week, end_of_prev_week),
                timestamp__time__range=(datetime.time(11, 15), datetime.time(13, 15)),
            )
            .annotate(weekday=ExtractWeekDay("timestamp"))
            .values("weekday")
            .annotate(count=Count("id"))
            .order_by("-count")
            .first()
        )
        if not top_day:
            return {
                "weekday": "Ingen data",
                "count": 0,
            }

        days = ["Søndag", "Mandag", "Tirsdag", "Onsdag", "Torsdag", "Fredag", "Lørdag"]
        return {
            "weekday": days[top_day["weekday"] - 1],
            "count": top_day["count"],
        }
