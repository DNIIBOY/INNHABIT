import json

from dashboard.utils import FakeMetadata
from django.db.models import Count, Max, Min, Model
from django.db.models.functions import ExtractMonth, ExtractYear
from django.http import HttpRequest, HttpResponse
from django.shortcuts import render
from django.utils import timezone
from django_components import Component, register
from occupancy.models import EntryEvent

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


def extract_counts_by_month(event_model: Model) -> dict[str, int]:
    groups = (
        event_model.objects.annotate(
            year=ExtractYear("timestamp"), month=ExtractMonth("timestamp")
        )
        .values("year", "month")
        .annotate(count=Count("id"))
        .order_by("year", "month")
    )

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
        print("lol")
        entry_stats = EntryEvent.objects.aggregate(
            earliest=Min("timestamp"), latest=Max("timestamp")
        )

        earliest_date = (
            entry_stats["earliest"] if entry_stats["earliest"] else timezone.now()
        )

        latest_date = entry_stats["latest"] if entry_stats["latest"] else timezone.now()

        start_year, start_month = earliest_date.year, earliest_date.month
        end_year, end_month = latest_date.year, latest_date.month

        entry_counts = extract_counts_by_month(EntryEvent)

        all_month_years = []
        labels = []

        current_year, current_month = start_year, start_month
        while (current_year < end_year) or (
            current_year == end_year and current_month <= end_month
        ):
            key = get_month_year_key(current_year, current_month)
            all_month_years.append(key)

            month_name = MONTHS[current_month - 1]
            if current_month == 1:
                labels.append(f"{month_name} {current_year}")
            else:
                labels.append(month_name)

            current_month += 1
            if current_month > 12:
                current_month = 1
                current_year += 1

        occupancy_counts = []
        for month_year in all_month_years:
            entries = entry_counts.get(month_year, 0)
            occupancy_counts.append(entries)

        context = {
            "labels": labels,
            "occupancy_counts": occupancy_counts,
        }

        return {
            **context,
            "json_data": json.dumps(context),
        }
