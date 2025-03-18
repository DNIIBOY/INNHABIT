from django.db.models import BooleanField, Value
from django_components import Component, register
from occupancy.models import EntryEvent, ExitEvent


@register("latest_events")
class LatestEvents(Component):
    template_name = "latest_events.html"

    def get_context_data(self, items: int = 4) -> dict:
        entry_events = EntryEvent.objects.annotate(
            is_entry=Value(True, output_field=BooleanField())
        ).prefetch_related("entrance")
        if not self.request.user.has_perm("occupancy.view_exit_event"):
            return {"events": entry_events.order_by("-timestamp")[:items]}

        exit_events = ExitEvent.objects.annotate(
            is_entry=Value(False, output_field=BooleanField())
        ).prefetch_related("entrance")
        if not self.request.user.has_perm("occupancy.view_entry_event"):
            return {"events": exit_events.order_by("-timestamp")[:items]}

        latest_events = entry_events.union(exit_events).order_by("-timestamp")[:items]
        return {
            "events": latest_events,
        }
