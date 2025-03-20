import json

from django.db.models import Count, F, Q
from django.utils import timezone
from django_components import Component, register
from occupancy.models import Entrance


@register("entrance_overview")
class EntranceOverview(Component):
    template_name = "entrance_overview.html"
    js_file = "entrance_overview.js"
    colors = ["#4066B2", "#5CAF8D", "#DF8E2E", "#CC445B"]

    class Media:
        js = ["https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.0.2/chart.min.js"]

    def get_context_data(self) -> dict:
        today = timezone.localtime().date()
        entrances = Entrance.objects.annotate(
            entry_count=Count(
                "entries", filter=Q(entries__timestamp__date=today), distinct=True
            ),
            exit_count=Count(
                "exits", filter=Q(entries__timestamp__date=today), distinct=True
            ),
            event_count=F("entry_count") + F("exit_count"),
        ).order_by("name")
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
