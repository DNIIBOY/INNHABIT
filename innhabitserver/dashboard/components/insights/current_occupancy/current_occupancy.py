from django.utils import timezone
from django_components import Component, register
from occupancy.models import EntryEvent, ExitEvent


@register("current_occupancy")
class CurrentOccupancy(Component):
    template_name = "current_occupancy.html"

    def get_context_data(self) -> dict:
        midnight = timezone.now().replace(hour=0, minute=0, second=0, microsecond=0)
        entries = EntryEvent.objects.filter(timestamp__gte=midnight).count()
        exits = ExitEvent.objects.filter(timestamp__gte=midnight).count()
        return {
            "current_occupancy": entries - exits,
        }
