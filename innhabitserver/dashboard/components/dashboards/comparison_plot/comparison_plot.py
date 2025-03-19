import json

import numpy as np
from django.db.models import Count, ExpressionWrapper, IntegerField
from django.db.models.functions import ExtractHour, ExtractMinute, Floor
from django.db.models.query import QuerySet
from django.utils import timezone
from django_components import Component, register
from occupancy.models import EntryEvent, ExitEvent


def extract_count_per_interval(queryset: QuerySet, interval: int) -> np.array:
    assert 60 % interval == 0, "Interval must be a divisor of 60"
    intervals_per_hour = 60 // interval
    groups = (
        queryset.annotate(
            hour=ExtractHour("timestamp"),
            minute_group=ExpressionWrapper(
                Floor((ExtractMinute("timestamp") / interval)),
                output_field=IntegerField(),
            ),
        )
        .values("hour", "minute_group")
        .annotate(count=Count("id"))
        .order_by("hour", "minute_group")
    )

    event_counts = np.zeros(24 * intervals_per_hour, dtype=int)
    for item in groups:
        event_counts[item["hour"] * intervals_per_hour + item["minute_group"]] = item[
            "count"
        ]
    return event_counts


@register("comparison_plot")
class ComparisonPlot(Component):
    template_name = "comparison_plot.html"
    js_file = "comparison_plot.js"

    class Media:
        js = ["https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.0.2/chart.min.js"]

    def get_context_data(self, interval: int = 30) -> dict:
        assert 60 % interval == 0, "Interval must be a divisor of 60"

        today = timezone.now().date()
        entries = extract_count_per_interval(
            EntryEvent.objects.filter(timestamp__date=today), interval
        )
        exits = extract_count_per_interval(
            ExitEvent.objects.filter(timestamp__date=today), interval
        )
        today_counts = entries.cumsum() - exits.cumsum()

        labels = []
        for hour in range(24):
            for minute_group in range(0, 60, interval):
                if minute_group == 0:
                    labels.append(f"{hour:02d}")
                else:
                    labels.append("")  # Empty labels for minutes

        context = {
            "labels": labels,
            "today_counts": today_counts.tolist(),
        }
        return {
            **context,
            "json_data": json.dumps(context),
        }
