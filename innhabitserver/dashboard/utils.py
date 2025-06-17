from datetime import date
from typing import Sequence

from django.contrib.auth.base_user import AbstractBaseUser
from django.core.exceptions import PermissionDenied
from django.db.models import BooleanField, CharField, Value
from django.db.models.query import QuerySet
from occupancy.models import (
    Entrance,
    EntryEvent,
    ExitEvent,
    TestEntryEvent,
    TestExitEvent,
)


def filter_events(
    user: AbstractBaseUser,
    entrances: Sequence[Entrance] | None = None,
    event_type: str | None = None,
    from_date: str | date | None = None,
    to_date: str | date | None = None,
    test_events: bool = False,
) -> QuerySet[EntryEvent | ExitEvent]:

    entry_model, exit_model = EntryEvent, ExitEvent
    if test_events:
        entry_model, exit_model = TestEntryEvent, TestExitEvent

    entry_events = entry_model.objects.annotate(
        is_entry=Value(True, output_field=BooleanField()),
        direction=Value("Ind", output_field=CharField()),
    ).prefetch_related("entrance")
    exit_events = exit_model.objects.annotate(
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
    if entrances:
        entry_events = entry_events.filter(entrance__in=entrances)
        exit_events = exit_events.filter(entrance__in=entrances)

    view_entry = user.has_perm("occupancy.view_entryevent")
    view_exit = user.has_perm("occupancy.view_exitevent")

    if view_entry and view_exit:
        events = entry_events.union(exit_events)
    elif view_entry and not view_exit:
        events = entry_events
    elif view_exit and not view_entry:
        events = exit_events
    else:
        raise PermissionDenied

    return events
