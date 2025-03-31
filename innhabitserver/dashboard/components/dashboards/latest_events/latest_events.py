import urllib.parse
from datetime import date
from typing import Sequence

from dashboard.utils import FakeMetadata, filter_events
from django.http import HttpRequest, HttpResponse
from django.shortcuts import render
from django_components import Component, register
from occupancy.forms import FilterEventsForm
from occupancy.models import Entrance


@register("latest_events")
class LatestEvents(Component):
    template_name = "latest_events.html"

    def get(self, request: HttpRequest) -> HttpResponse:
        filters = FilterEventsForm(request.GET)
        if not filters.is_valid():
            return HttpResponse(filters.errors.items(), status=400)

        kwargs = {
            **filters.cleaned_data,
            "items": int(request.GET.get("items", 4)),
            "offset": int(request.GET.get("offset", 0)),
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
        event_type: str = "all",
        from_date: date | None = None,
        to_date: date | None = None,
        offset: int = 0,
        entrances: Sequence[Entrance] | Sequence[int] | Entrance | int | None = None,
        infinite_scroll: bool = False,
        hide_title: bool = False,
        timestamp_format: str = "H:i",
    ) -> dict:
        events = filter_events(
            user=self.request.user,
            entrances=entrances,
        )
        latest_events = events.order_by("-timestamp")[offset : offset + items]

        if isinstance(entrances, Sequence):
            entrances = list(
                set(
                    entrance.id if isinstance(entrance, Entrance) else entrance
                    for entrance in entrances
                )
            )
        elif isinstance(entrances, Entrance):
            entrances = entrances.id

        kwargs = {
            "items": items,
            "offset": offset + items,
            "entrances": entrances if entrances else [],
            "infinite_scroll": infinite_scroll,
            "hide_title": hide_title,
            "timestamp_format": timestamp_format,
        }
        scroll_params = urllib.parse.urlencode(kwargs, True)
        kwargs["offset"] = offset

        return {
            "params": urllib.parse.urlencode(kwargs, True),
            "scroll_params": scroll_params,
            "events": latest_events,
            **kwargs,
        }
