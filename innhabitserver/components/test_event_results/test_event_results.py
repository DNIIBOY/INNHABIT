import numpy as np
from django.db.models import BooleanField, Value
from django.db.models.query import QuerySet
from django.utils import timezone
from django_components import Component, register
from occupancy.models import EntryEvent, ExitEvent, TestEntryEvent, TestExitEvent
from scipy.optimize import linear_sum_assignment

MAX_TIME_DIFF = 12  # Max between real event and test event


def hungarian_event_map(
    system_events: QuerySet[EntryEvent | ExitEvent],
    manual_events: QuerySet[TestEntryEvent | TestExitEvent],
    max_time_diff: int = 10,
):
    grouped_system_events: dict[str, list[EntryEvent | ExitEvent]] = {}
    grouped_manual_events: dict[str, list[TestEntryEvent | TestExitEvent]] = {}

    for event in system_events:
        if event.entrance not in grouped_system_events:
            grouped_system_events[event.entrance] = []
        grouped_system_events[event.entrance].append(event)

    for event in manual_events:
        if event.entrance not in grouped_manual_events:
            grouped_manual_events[event.entrance] = []
        grouped_manual_events[event.entrance].append(event)

    results = []
    for entrance, system_events_in_group in grouped_system_events.items():
        if entrance not in grouped_manual_events:
            continue
        manual_events_in_group = grouped_manual_events[entrance]

        unmapped_manual_events = set(manual_events_in_group)
        unmapped_system_events = set(system_events_in_group)

        n_system = len(system_events_in_group)
        n_manual = len(manual_events_in_group)
        INF = float("inf")

        cost_matrix = np.full((n_manual, n_system), INF)

        for i, manual_event in enumerate(manual_events_in_group):
            for j, system_events in enumerate(system_events_in_group):
                time_diff = abs(
                    (manual_event.timestamp - system_events.timestamp).total_seconds()
                )
                if time_diff <= max_time_diff:
                    cost_matrix[i, j] = time_diff

        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        for row_idx, col_idx in zip(row_indices, col_indices):
            if cost_matrix[row_idx, col_idx] == INF:
                continue
            manual_event = manual_events_in_group[row_idx]
            sys_event = system_events_in_group[col_idx]
            unmapped_manual_events.remove(manual_event)
            unmapped_system_events.remove(sys_event)
            results.append(
                {
                    "timestamp": sys_event.timestamp,
                    "entrance": sys_event.entrance,
                    "system_entry": sys_event.is_entry,
                    "manual_entry": manual_event.is_entry,
                    "is_equal": sys_event.is_entry == manual_event.is_entry,
                }
            )

        for sys_event in unmapped_system_events:
            results.append(
                {
                    "timestamp": sys_event.timestamp,
                    "entrance": sys_event.entrance,
                    "system_entry": sys_event.is_entry,
                    "manual_entry": None,
                    "is_equal": False,
                }
            )
        for man_event in unmapped_manual_events:
            results.append(
                {
                    "timestamp": man_event.timestamp,
                    "entrance": man_event.entrance,
                    "system_entry": None,
                    "manual_entry": man_event.is_entry,
                    "is_equal": False,
                }
            )
    results.sort(key=lambda x: x["timestamp"])
    return results


def map_events(
    system_events: QuerySet[EntryEvent | ExitEvent],
    manual_events: QuerySet[EntryEvent | ExitEvent],
) -> list[dict]:
    system_events = set(system_events)
    manual_events = set(manual_events)

    results = []

    for sys_event in system_events:
        # Find closest manual event within 5 seconds and same entrance
        closest_manual = None
        min_time_diff = MAX_TIME_DIFF

        for man_event in manual_events:
            if man_event.entrance != sys_event.entrance:
                continue  # Only match if entrance is the same

            time_diff = abs((sys_event.timestamp - man_event.timestamp).total_seconds())
            if time_diff <= MAX_TIME_DIFF and time_diff <= min_time_diff:
                closest_manual = man_event
                min_time_diff = time_diff

        # Create result entry
        if closest_manual:
            manual_events.remove(closest_manual)
            results.append(
                {
                    "timestamp": sys_event.timestamp,
                    "entrance": sys_event.entrance,
                    "system_entry": sys_event.is_entry,
                    "manual_entry": closest_manual.is_entry,
                    "is_equal": sys_event.is_entry == closest_manual.is_entry,
                }
            )
        else:
            results.append(
                {
                    "timestamp": sys_event.timestamp,
                    "entrance": sys_event.entrance,
                    "system_entry": sys_event.is_entry,
                    "manual_entry": None,
                    "is_equal": False,
                }
            )

    for man_event in manual_events:
        results.append(
            {
                "timestamp": man_event.timestamp,
                "entrance": man_event.entrance,
                "system_entry": None,
                "manual_entry": man_event.is_entry,
                "is_equal": False,
            }
        )

    results.sort(key=lambda x: x["timestamp"])
    return results


def get_test_results() -> list[dict]:
    today = timezone.localdate()
    today_test_events = TestEntryEvent.objects.filter(timestamp__date=today).union(
        TestExitEvent.objects.filter(timestamp__date=today)
    )
    first_event = today_test_events.order_by("timestamp").first()
    last_event = today_test_events.order_by("-timestamp").first()
    if not first_event or not last_event:
        return []
    time_range = (first_event.timestamp, last_event.timestamp)
    entry_events = (
        EntryEvent.objects.filter(timestamp__range=time_range)
        .annotate(
            is_entry=Value(True, output_field=BooleanField()),
        )
        .prefetch_related("entrance")
    )
    exit_events = (
        ExitEvent.objects.filter(timestamp__range=time_range)
        .annotate(
            is_entry=Value(False, output_field=BooleanField()),
        )
        .prefetch_related("entrance")
    )
    test_entry_events = (
        TestEntryEvent.objects.filter(timestamp__range=time_range)
        .annotate(
            is_entry=Value(True, output_field=BooleanField()),
        )
        .prefetch_related("entrance")
    )
    test_exit_events = (
        TestExitEvent.objects.filter(timestamp__range=time_range)
        .annotate(
            is_entry=Value(False, output_field=BooleanField()),
        )
        .prefetch_related("entrance")
    )
    system_events = entry_events.union(exit_events)
    manual_events = test_entry_events.union(test_exit_events)

    return map_events(system_events, manual_events)


@register("test_event_results")
class TestEventResults(Component):
    template_name = "test_event_results.html"

    def get_context_data(self) -> dict:
        return {"events": get_test_results()}
