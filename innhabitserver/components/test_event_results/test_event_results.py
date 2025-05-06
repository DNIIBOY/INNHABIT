from django.db.models import BooleanField, Value
from django.db.models.query import QuerySet
from django.utils import timezone
from django_components import Component, register
from occupancy.models import EntryEvent, ExitEvent, TestEntryEvent, TestExitEvent

MAX_TIME_DIFF = 12  # Max between real event and test event


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


@register("test_event_results")
class TestEventResults(Component):
    template_name = "test_event_results.html"

    def get_context_data(self) -> dict:
        today = timezone.localdate()
        today_test_events = TestEntryEvent.objects.filter(timestamp__date=today).union(
            TestExitEvent.objects.filter(timestamp__date=today)
        )
        first_event = today_test_events.order_by("timestamp").first()
        last_event = today_test_events.order_by("-timestamp").first()
        if not first_event or not last_event:
            return {"events": []}
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

        events = map_events(system_events, manual_events)
        return {"events": events}
