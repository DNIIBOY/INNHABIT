from django.db.models import BooleanField, Value
from django.http import HttpRequest, HttpResponse
from django.shortcuts import get_object_or_404
from django_components import Component, register
from occupancy.models import Entrance, EntryEvent, ExitEvent


@register("latest_events")
class LatestEvents(Component):
    template_name = "latest_events.html"

    def get(self, request: HttpRequest) -> HttpResponse:
        items = int(request.GET.get("items", 4))
        entrance = request.GET.get("entrance")
        entrance = int(entrance) if entrance else None
        kwargs = {"items": items, "entrance": entrance}

        return self.render_to_response(request=request, kwargs=kwargs)

    def get_context_data(
        self, items: int = 4, entrance: Entrance | int | None = None
    ) -> dict:
        if isinstance(entrance, int):
            entrance = get_object_or_404(Entrance, id=entrance)
        entry_events = EntryEvent.objects.annotate(
            is_entry=Value(True, output_field=BooleanField())
        ).prefetch_related("entrance")
        if entrance:
            entry_events = entry_events.filter(entrance=entrance)

        exit_events = ExitEvent.objects.annotate(
            is_entry=Value(False, output_field=BooleanField())
        ).prefetch_related("entrance")
        if entrance:
            exit_events = exit_events.filter(entrance=entrance)

        view_entry = self.request.user.has_perm("occupancy.view_entry_event")
        view_exit = self.request.user.has_perm("occupancy.view_exit_event")

        if view_entry and view_exit:
            events = entry_events.union(exit_events)
        elif view_entry and not view_exit:
            events = entry_events
        elif view_exit and not view_entry:
            events = exit_events
        else:
            events = EntryEvent.objects.none()

        events = entry_events.union(exit_events)
        latest_events = events.order_by("-timestamp")[:items]

        return {
            "events": latest_events,
        }
