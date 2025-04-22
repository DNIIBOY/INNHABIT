from django.db.models import BooleanField, Value
from django.utils import timezone
from django_components import Component, register
from occupancy.models import EntryEvent, ExitEvent, TestEntryEvent, TestExitEvent


def map_events(system_events, manual_events):
    # Convert union querysets to lists first to avoid the "after union()" limitation
    system_events_list = list(system_events)
    manual_events_list = list(manual_events)

    results = []
    matched_manual_ids = set()

    # Process system events first
    for sys_event in system_events_list:
        # Find closest manual event within 5 seconds and same entrance
        closest_manual = None
        min_time_diff = 5.01  # Slightly more than 5 seconds

        for man_event in manual_events_list:
            if man_event.id in matched_manual_ids:
                continue

            if man_event.entrance != sys_event.entrance:
                continue  # Only match if entrance is the same

            time_diff = abs((sys_event.timestamp - man_event.timestamp).total_seconds())
            if time_diff <= 5 and time_diff < min_time_diff:
                closest_manual = man_event
                min_time_diff = time_diff

        # Create result entry
        if closest_manual:
            matched_manual_ids.add(closest_manual.id)
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

    # Add unmatched manual events
    for man_event in manual_events_list:
        if man_event.id not in matched_manual_ids:
            results.append(
                {
                    "timestamp": man_event.timestamp,
                    "entrance": man_event.entrance,
                    "system_entry": None,
                    "manual_entry": man_event.is_entry,
                    "is_equal": False,
                }
            )

    # Sort results by timestamp
    results.sort(key=lambda x: x["timestamp"])
    return results


@register("test_event_results")
class TestEventResults(Component):
    template_name = "test_event_results.html"

    def get_context_data(self) -> dict:
        today = timezone.localdate()
        entry_events = (
            EntryEvent.objects.filter(
                timestamp__date=today,
            )
            .annotate(
                is_entry=Value(True, output_field=BooleanField()),
            )
            .prefetch_related("entrance")
        )
        exit_events = (
            ExitEvent.objects.filter(
                timestamp__date=today,
            )
            .annotate(
                is_entry=Value(False, output_field=BooleanField()),
            )
            .prefetch_related("entrance")
        )
        test_entry_events = (
            TestEntryEvent.objects.filter(
                timestamp__date=today,
            )
            .annotate(
                is_entry=Value(True, output_field=BooleanField()),
            )
            .prefetch_related("entrance")
        )
        test_exit_events = (
            TestExitEvent.objects.filter(
                timestamp__date=today,
            )
            .annotate(
                is_entry=Value(False, output_field=BooleanField()),
            )
            .prefetch_related("entrance")
        )
        system_events = entry_events.union(exit_events)
        manual_events = test_entry_events.union(test_exit_events)

        events = map_events(system_events, manual_events)
        print(events)
        return {"events": events}
