"""
This is very much vibe code where comparison_plot was converted to an all time thing
"""

import json

from dashboard.utils import FakeMetadata
from django.db.models import Count, Max, Min, Model
from django.db.models.functions import ExtractMonth, ExtractYear
from django.db.models.query import QuerySet
from django.http import HttpRequest, HttpResponse
from django.shortcuts import render
from django.utils import timezone
from django_components import Component, register
from occupancy.models import EntryEvent, ExitEvent

MONTHS = [
    "Januar",
    "Februar",
    "Marts",
    "April",
    "Maj",
    "Juni",
    "Juli",
    "August",
    "September",
    "Oktober",
    "November",
    "December",
]


def get_month_year_key(year: int, month: str) -> str:
    return f"{year}-{month:02d}"


def extract_counts_by_month(event_model: Model) -> QuerySet:
    groups = (
        event_model.objects.annotate(
            year=ExtractYear("timestamp"), month=ExtractMonth("timestamp")
        )
        .values("year", "month")
        .annotate(count=Count("id"))
        .order_by("year", "month")
    )

    # Build a dictionary of counts by year-month
    event_counts = {}
    for item in groups:
        key = get_month_year_key(item["year"], item["month"])
        event_counts[key] = item["count"]

    return event_counts


@register("all_time_plot")
class AllTimePlot(Component):
    template_name = "all_time_plot.html"
    js_file = "all_time_plot.js"

    class Media:
        js = ["https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.0.2/chart.min.js"]

    def get(self, request: HttpRequest) -> HttpResponse:
        if not request.htmx:
            return self.render_to_response(request=request)

        with self._with_metadata(FakeMetadata(request)):
            context = self.get_context_data()

        return render(request, self.template_name + "#json_element", context)

    def get_context_data(self) -> dict:
        # Get the earliest and latest timestamps from both event types
        entry_stats = EntryEvent.objects.aggregate(
            earliest=Min("timestamp"), latest=Max("timestamp")
        )
        exit_stats = ExitEvent.objects.aggregate(
            earliest=Min("timestamp"), latest=Max("timestamp")
        )

        # Determine overall earliest and latest dates
        earliest_date = (
            min(
                entry_stats["earliest"] or timezone.now(),
                exit_stats["earliest"] or timezone.now(),
            )
            if exit_stats["earliest"] or entry_stats["earliest"]
            else timezone.now()
        )

        latest_date = (
            max(
                entry_stats["latest"] or timezone.now(),
                exit_stats["latest"] or timezone.now(),
            )
            if exit_stats["latest"] or entry_stats["latest"]
            else timezone.now()
        )

        # Extract the year and month from earliest and latest dates
        start_year, start_month = earliest_date.year, earliest_date.month
        end_year, end_month = latest_date.year, latest_date.month

        # Get entry and exit counts for all time periods
        entry_counts = extract_counts_by_month(EntryEvent)
        exit_counts = extract_counts_by_month(ExitEvent)

        # Generate all month-year combinations between earliest and latest
        all_month_years = []
        labels = []

        current_year, current_month = start_year, start_month
        while (current_year < end_year) or (
            current_year == end_year and current_month <= end_month
        ):
            key = get_month_year_key(current_year, current_month)
            all_month_years.append(key)

            # Create more readable labels
            month_name = MONTHS[current_month - 1]
            if current_month == 1:  # Only show year for January
                labels.append(f"{month_name} {current_year}")
            else:
                labels.append(month_name)

            # Move to next month
            current_month += 1
            if current_month > 12:
                current_month = 1
                current_year += 1

        # Calculate cumulative occupancy over time
        occupancy_counts = []

        for month_year in all_month_years:
            entries = entry_counts.get(month_year, 0)
            exits = exit_counts.get(month_year, 0)

            occupancy_counts.append(entries - exits)

        context = {
            "labels": labels,
            "occupancy_counts": occupancy_counts,
        }

        return {
            **context,
            "json_data": json.dumps(context),
        }
