from django.utils import timezone
from django_components import Component, register
from occupancy.models import EntryEvent


@register("visitors_today")
class VisitorsToday(Component):
    template_name = "visitors_today.html"

    def get_context_data(self) -> dict:
        today = timezone.localtime().date()
        entries = EntryEvent.objects.filter(timestamp__date=today).count()
        return {
            "visitors_today": entries,
        }
