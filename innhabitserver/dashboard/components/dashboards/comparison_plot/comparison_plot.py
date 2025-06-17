import json
from datetime import timedelta
from typing import Any

import numpy as np
from django.db.models import Count, ExpressionWrapper, IntegerField, Q
from django.db.models.functions import ExtractHour, ExtractMinute, Floor
from django.db.models.query import QuerySet
from django.http import HttpRequest, HttpResponse
from django.shortcuts import render
from django.template.context import Context
from django.utils import timezone
from django_components import Component, register
from occupancy.models import EntryEvent, ExitEvent

WEEKDAYS = [
    "Mandag",
    "Tirsdag",
    "Onsdag",
    "Torsdag",
    "Fredag",
    "Lørdag",
    "Søndag",
]


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

    def get(self, request: HttpRequest) -> HttpResponse:
        if not request.htmx:
            return self.render_to_response(request=request)

        self.request = request
        context = self.get_context_data()
        return render(request, self.template_name + "#json_element", context)

    def get_template_data(
        self, args: Any, kwargs: Any, slots: Any, context: Context
    ) -> dict:
        interval = int(kwargs.get("interval", 10))
        assert 60 % interval == 0, "Interval must be a divisor of 60"

        now = timezone.localtime()
        today = now.date()
        today_entries = extract_count_per_interval(
            EntryEvent.objects.filter(timestamp__date=today), interval
        )
        today_exits = extract_count_per_interval(
            ExitEvent.objects.filter(timestamp__date=today), interval
        )
        today_counts = today_entries.cumsum() - today_exits.cumsum()

        prev_month = today.replace(day=1) - timedelta(days=1)
        weekday_entries = extract_count_per_interval(
            EntryEvent.objects.filter(
                ~Q(timestamp__date=today),
                timestamp__year__gte=prev_month.year,
                timestamp__month__gte=prev_month.month,
                timestamp__week_day=(today.weekday() + 2) % 7,
            ),
            interval,
        )
        weekday_exits = extract_count_per_interval(
            ExitEvent.objects.filter(
                ~Q(timestamp__date=today),
                timestamp__year__gte=prev_month.year,
                timestamp__month__gte=prev_month.month,
                timestamp__week_day=(today.weekday() + 2) % 7,
            ),
            interval,
        )

        weekday_count = 0
        date = prev_month
        while date < today:
            if date.weekday() == today.weekday():
                weekday_count += 1
            date += timedelta(days=1)

        avg_weekday_counts = (
            weekday_entries.cumsum() - weekday_exits.cumsum()
        ) // weekday_count

        labels: list[str] = []
        current_time = 0
        for hour in range(24):
            for minute_group in range(0, 60, interval):
                if (
                    now.hour == hour
                    and now.minute // interval == minute_group // interval
                ):
                    current_time = len(labels)
                if minute_group == 0:
                    labels.append(f"{hour:02d}")
                else:
                    labels.append("")  # Empty labels for minutes

        context = {
            "weekday": WEEKDAYS[today.weekday()],
            "labels": labels,
            "today_counts": today_counts.tolist()[: current_time + 1],
            "avg_weekday_counts": avg_weekday_counts.tolist(),
        }
        return {
            **context,
            "json_data": json.dumps(context),
        }
