from typing import Any

from django.template.context import Context
from django.utils import timezone
from django_components import Component, register
from occupancy.models import EntryEvent


@register("visitors_today")
class VisitorsToday(Component):
    template_name = "visitors_today.html"

    def get_template_data(
        self, args: Any, kwargs: Any, slots: Any, context: Context
    ) -> dict:
        today = timezone.localtime().date()
        entries = EntryEvent.objects.filter(timestamp__date=today).count()
        return {
            "visitors_today": entries,
        }
