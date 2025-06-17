import urllib.parse
from datetime import date
from typing import Any, NamedTuple, Sequence

from dashboard.utils import filter_events
from django.db.models import QuerySet
from django.forms import BooleanField
from django.http import HttpRequest, HttpResponse
from django.shortcuts import render
from django.template.context import Context
from django_components import Component, Default, register
from occupancy.forms import FilterEventsForm
from occupancy.models import Entrance, EntryEvent, ExitEvent


@register("latest_events")
class LatestEvents(Component):
    template_name = "latest_events.html"

    class Kwargs(NamedTuple):
        items: int
        event_type: str
        from_date: date | None
        to_date: date | None
        offset: int
        entrances: Sequence[Entrance] | None
        infinite_scroll: bool
        hide_title: bool
        timestamp_format: str
        test_events: bool

    class Defaults:
        items: int = 4
        event_type: str = "all"
        from_date: date | None = None
        to_date: date | None = None
        offset: int = 0
        entrances: Sequence[Entrance] | None = Default(lambda: [])
        infinite_scroll: bool = False
        hide_title: bool = False
        timestamp_format: str = "H:i"
        test_events: bool = False

    class View:
        def get(self, request: HttpRequest) -> HttpResponse:
            filters = FilterEventsForm(request.GET)
            if not filters.is_valid():
                return HttpResponse(filters.errors.items(), status=400)

            kwargs = {
                **filters.cleaned_data,
                "items": int(request.GET.get("items", 4)),
                "offset": int(request.GET.get("offset", 0)),
                "infinite_scroll": BooleanField(initial=False).to_python(
                    request.GET.get("infinite_scroll")
                ),
                "hide_title": BooleanField(initial=False).to_python(
                    request.GET.get("hide_title")
                ),
                "timestamp_format": request.GET.get("timestamp_format", "H:i"),
            }
            if not request.htmx:
                return self.component_cls.render_to_response(
                    request=request, kwargs=kwargs
                )

            # events = get_latest_events(request, self.component_cls.Kwargs(**kwargs))
            context = self.component_cls.get_template_data(
                args=None,
                kwargs=self.component_cls.Kwargs(**kwargs),
                slots=None,
                context=None,
            )
            return render(
                request,
                self.component_cls.template_name + "#event_rows",
            )

    def get_template_data(
        self, args: Any, kwargs: Kwargs, slots: Any, context: Context
    ) -> dict:
        assert isinstance(
            kwargs, self.Kwargs
        ), "kwargs must be of type LatestEvents.Kwargs"
        items = kwargs.items
        offset = kwargs.offset
        latest_events = get_latest_events(self.request, kwargs)

        entrances = kwargs.entrances
        if isinstance(entrances, Sequence):
            entrances = list(
                set(
                    entrance.id if isinstance(entrance, Entrance) else entrance
                    for entrance in entrances
                )
            )
        elif isinstance(entrances, Entrance):
            entrances = entrances.id

        new_kwargs = {
            "items": items,
            "offset": offset + items,
            "entrances": entrances,
            "infinite_scroll": kwargs.infinite_scroll,
            "hide_title": kwargs.hide_title,
            "timestamp_format": kwargs.timestamp_format,
        }
        scroll_params = urllib.parse.urlencode(new_kwargs, True)
        new_kwargs["offset"] = offset

        return {
            "params": urllib.parse.urlencode(new_kwargs, True),
            "scroll_params": scroll_params,
            "events": latest_events,
            **self.raw_kwargs,
        }


def get_latest_events(
    request: HttpRequest, kwargs: LatestEvents.Kwargs
) -> QuerySet[EntryEvent | ExitEvent]:
    offset = kwargs.offset
    items = kwargs.items
    events = filter_events(
        user=request.user,
        entrances=kwargs.entrances,
        event_type=kwargs.event_type,
        from_date=kwargs.from_date,
        to_date=kwargs.to_date,
        test_events=kwargs.test_events,
    )
    latest_events = events.order_by("-timestamp")[offset : offset + items]
    return latest_events
