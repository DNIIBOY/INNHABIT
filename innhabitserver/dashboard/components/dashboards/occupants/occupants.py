from datetime import timedelta
from typing import Any

from django.http import HttpRequest, HttpResponse
from django.template.context import Context
from django.utils import timezone
from django_components import Component, register
from occupancy.models import EntryEvent, ExitEvent


@register("occupants")
class Occupants(Component):
    template_name = "occupants.html"

    def get(self, request: HttpRequest) -> HttpResponse:
        return self.render_to_response(request=request)

    def get_template_data(
        self, args: Any, kwargs: Any, slots: Any, context: Context
    ) -> dict:
        now = timezone.localtime()
        comparison_time = now - timedelta(hours=1)
        today = now.date()
        entries = EntryEvent.objects.filter(timestamp__date=today).count()
        exits = ExitEvent.objects.filter(timestamp__date=today).count()

        comparsion_entries = EntryEvent.objects.filter(
            timestamp__gte=comparison_time
        ).count()
        comparsion_exits = ExitEvent.objects.filter(
            timestamp__gte=comparison_time
        ).count()
        return {
            "current_occupancy": entries - exits,
            "comparison_occupancy": comparsion_entries - comparsion_exits,
        }
