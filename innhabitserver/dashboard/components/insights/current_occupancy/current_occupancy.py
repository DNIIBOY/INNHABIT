from django.utils import timezone
from django_components import Component, register
from occupancy.models import EntryEvent, ExitEvent


@register("current_occupancy")
class CurrentOccupancy(Component):
    template_name = "current_occupancy.html"

    def get_context_data(self) -> dict:
        today = timezone.now().date()
        entries = EntryEvent.objects.filter(timestamp__date=today).count()
        exits = ExitEvent.objects.filter(timestamp__date=today).count()
        return {
            "current_occupancy": entries - exits,
        }
