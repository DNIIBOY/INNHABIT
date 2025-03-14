from django.utils import timezone
from django_components import Component, register
from occupancy.models import EntryEvent, ExitEvent


@register("occupants")
class Occupants(Component):
    template_name = "occupants.html"

    def get_context_data(self) -> dict:
        today = timezone.now().date()
        entries = EntryEvent.objects.filter(timestamp__date=today).count()
        exits = ExitEvent.objects.filter(timestamp__date=today).count()
        return {"humans": entries - exits}
