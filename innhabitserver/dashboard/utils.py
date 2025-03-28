from dataclasses import dataclass
from datetime import date
from typing import Sequence

from django.contrib.auth.base_user import AbstractBaseUser
from django.core.exceptions import PermissionDenied
from django.db.models import BooleanField, CharField, Value
from django.db.models.query import QuerySet
from django.http import HttpRequest
from occupancy.models import Entrance, EntryEvent, ExitEvent


@dataclass
class FakeMetadata:
    request: HttpRequest


def filter_events(
    user: AbstractBaseUser,
    entrances: Sequence[Entrance] | Sequence[int] | Entrance | int | None = None,
    event_type: str | None = None,
    from_date: str | date | None = None,
    to_date: str | date | None = None,
) -> QuerySet[EntryEvent | ExitEvent]:
    entrance_ids = None
    if entrances:
        if not isinstance(entrances, Sequence):
            entrances = [entrances]
        entrance_ids = set(
            entrance.id if isinstance(entrance, Entrance) else entrance
            for entrance in entrances
        )

    entry_events = EntryEvent.objects.annotate(
        is_entry=Value(True, output_field=BooleanField()),
        direction=Value("Ind", output_field=CharField()),
    ).prefetch_related("entrance")
    exit_events = ExitEvent.objects.annotate(
        is_entry=Value(False, output_field=BooleanField()),
        direction=Value("Ud", output_field=CharField()),
    ).prefetch_related("entrance")

    if event_type == "entry":
        exit_events = exit_events.none()
    if event_type == "exit":
        entry_events = entry_events.none()

    if from_date:
        entry_events = entry_events.filter(timestamp__date__gte=from_date)
        exit_events = exit_events.filter(timestamp__date__gte=from_date)
    if to_date:
        entry_events = entry_events.filter(timestamp__date__lte=to_date)
        exit_events = exit_events.filter(timestamp__date__lte=to_date)
    if entrance_ids:
        entry_events = entry_events.filter(entrance__id__in=entrance_ids)
        exit_events = exit_events.filter(entrance__id__in=entrance_ids)

    view_entry = user.has_perm("occupancy.view_entry_event")
    view_exit = user.has_perm("occupancy.view_exit_event")

    if view_entry and view_exit:
        events = entry_events.union(exit_events)
    elif view_entry and not view_exit:
        events = entry_events
    elif view_exit and not view_entry:
        events = exit_events
    else:
        raise PermissionDenied

    return events
