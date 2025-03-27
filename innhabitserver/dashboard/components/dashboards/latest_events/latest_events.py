import urllib.parse

from dashboard.utils import FakeMetadata
from django.db.models import BooleanField, Value
from django.http import HttpRequest, HttpResponse
from django.shortcuts import get_object_or_404, render
from django_components import Component, register
from occupancy.models import Entrance, EntryEvent, ExitEvent


@register("latest_events")
class LatestEvents(Component):
    template_name = "latest_events.html"

    def get(self, request: HttpRequest) -> HttpResponse:
        entrance = request.GET.get("entrance")
        entrance = int(entrance) if entrance else None
        kwargs = {
            "items": int(request.GET.get("items", 4)),
            "offset": int(request.GET.get("offset", 0)),
            "entrance": entrance,
            "infinite_scroll": request.GET.get("infinite_scroll", "False").lower()
            == "true",
            "hide_title": request.GET.get("hide_title", "False").lower() == "true",
            "timestamp_format": request.GET.get("timestamp_format", "H:i"),
        }

        if not request.htmx:
            return self.render_to_response(request=request, kwargs=kwargs)

        with self._with_metadata(FakeMetadata(request)):
            context = self.get_context_data(**kwargs)
        return render(request, self.template_name + "#event_rows", context)

    def get_context_data(
        self,
        items: int = 4,
        offset: int = 0,
        entrance: Entrance | int | None = None,
        infinite_scroll: bool = False,
        hide_title: bool = False,
        timestamp_format: str = "H:i",
    ) -> dict:
        if isinstance(entrance, int):
            entrance = get_object_or_404(Entrance, id=entrance)
        entry_events = EntryEvent.objects.annotate(
            is_entry=Value(True, output_field=BooleanField())
        ).prefetch_related("entrance")

        exit_events = ExitEvent.objects.annotate(
            is_entry=Value(False, output_field=BooleanField())
        ).prefetch_related("entrance")

        if entrance:
            entry_events = entry_events.filter(entrance=entrance)
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

        latest_events = events.order_by("-timestamp")[offset : offset + items]

        kwargs = {
            "items": items,
            "offset": offset + items,
            "entrance": entrance.pk if entrance else "",  # type: ignore[union-attr]
            "infinite_scroll": infinite_scroll,
            "hide_title": hide_title,
            "timestamp_format": timestamp_format,
        }
        scroll_params = urllib.parse.urlencode(kwargs)
        kwargs["offset"] = offset

        return {
            "params": urllib.parse.urlencode(kwargs),
            "scroll_params": scroll_params,
            "events": latest_events,
            **kwargs,
        }
