import json

from django.db.models import Count, F, IntegerField, OuterRef, Subquery, Value
from django.db.models.functions import Coalesce
from django.http import HttpRequest, HttpResponse
from django.shortcuts import render
from django.utils import timezone
from django_components import Component, register
from occupancy.models import Entrance, EntryEvent, ExitEvent


@register("entrance_overview")
class EntranceOverview(Component):
    template_name = "entrance_overview.html"
    js_file = "entrance_overview.js"
    colors = ["#4066B2", "#5CAF8D", "#DF8E2E", "#CC445B"]

    class Media:
        js = ["https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.0.2/chart.min.js"]

    def get(self, request: HttpRequest) -> HttpResponse:
        if not request.htmx:
            return self.render_to_response(request=request)

        self.request = request
        context = self.get_context_data()
        return render(request, self.template_name + "#json_element", context)

    def get_context_data(self) -> dict:
        today = timezone.localtime().date()
        entry_counts = (
            EntryEvent.objects.filter(entrance=OuterRef("pk"), timestamp__date=today)
            .values("entrance")
            .annotate(count=Count("id"))
            .values("count")
        )

        exit_counts = (
            ExitEvent.objects.filter(entrance=OuterRef("pk"), timestamp__date=today)
            .values("entrance")
            .annotate(count=Count("id"))
            .values("count")
        )

        entrances = (
            Entrance.objects.annotate(
                entry_count=Coalesce(
                    Subquery(entry_counts, output_field=IntegerField()), Value(0)
                ),
                exit_count=Coalesce(
                    Subquery(exit_counts, output_field=IntegerField()), Value(0)
                ),
            )
            .annotate(event_count=F("entry_count") + F("exit_count"))
            .order_by("name")
        )

        labels = []
        events = []
        for entrance in entrances:
            labels.append(entrance.name)
            events.append(entrance.event_count)
        context = {
            "labels": labels,
            "events": events,
            "colors": self.colors[: len(labels)],
            "label_colors": tuple(zip(labels, self.colors[: len(labels)])),
        }
        return {
            **context,
            "json_data": json.dumps(context),
        }
