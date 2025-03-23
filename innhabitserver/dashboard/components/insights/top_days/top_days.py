import random

from dashboard.models import LabelledDate
from django.db.models import Count
from django.db.models.functions import TruncDate
from django_components import Component, register
from occupancy.models import EntryEvent


@register("top_days")
class TopDays(Component):
    template_name = "top_days.html"

    def get_context_data(self) -> dict:
        days_with_most_entries = (
            EntryEvent.objects.annotate(date=TruncDate("timestamp"))
            .values("date")
            .annotate(entry_count=Count("id"))
            .order_by("-entry_count")
            .values("date", "entry_count")[:3]
        )
        day = random.choice(days_with_most_entries)
        label = LabelledDate.objects.filter(date=day["date"]).first()

        return {
            "date": day["date"],
            "count": day["entry_count"],
            "label": label.label if label else None,
        }
