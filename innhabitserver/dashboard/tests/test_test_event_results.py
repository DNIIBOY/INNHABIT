from datetime import UTC, datetime, timedelta

from components.test_event_results.test_event_results import map_events
from django.db.models import BooleanField, Value
from django.test import TestCase
from occupancy.models import (
    Entrance,
    EntryEvent,
    ExitEvent,
    TestEntryEvent,
    TestExitEvent,
)


class TestMapEvents(TestCase):
    def setUp(self) -> None:
        self.entrances = [
            Entrance.objects.create(name="Entrance 1"),
            Entrance.objects.create(name="Entrance 2"),
        ]
        self.base_time = datetime(2000, 1, 1, tzinfo=UTC)
        for i in range(3):
            EntryEvent.objects.create(
                entrance=self.entrances[0],
                timestamp=self.base_time + timedelta(minutes=i),
            )

        for i in range(3, 6):
            ExitEvent.objects.create(
                entrance=self.entrances[0],
                timestamp=self.base_time + timedelta(minutes=i),
            )

        entry_events = (
            EntryEvent.objects.all()
            .annotate(
                is_entry=Value(True, output_field=BooleanField()),
            )
            .prefetch_related("entrance")
        )
        exit_events = (
            ExitEvent.objects.all()
            .annotate(
                is_entry=Value(False, output_field=BooleanField()),
            )
            .prefetch_related("entrance")
        )
        self.system_events = entry_events.union(exit_events)

    def _get_mapped(self) -> list[dict]:
        test_entry_events = (
            TestEntryEvent.objects.all()
            .annotate(
                is_entry=Value(True, output_field=BooleanField()),
            )
            .prefetch_related("entrance")
        )
        test_exit_events = (
            TestExitEvent.objects.all()
            .annotate(
                is_entry=Value(False, output_field=BooleanField()),
            )
            .prefetch_related("entrance")
        )
        manual_events = test_entry_events.union(test_exit_events)
        return map_events(self.system_events, manual_events)

    def test_all_match(self) -> None:
        for i in range(3):
            TestEntryEvent.objects.create(
                entrance=self.entrances[0],
                timestamp=self.base_time + timedelta(minutes=i),
            )
        for i in range(3, 6):
            TestExitEvent.objects.create(
                entrance=self.entrances[0],
                timestamp=self.base_time + timedelta(minutes=i),
            )

        result = self._get_mapped()
        self.assertSequenceEqual(
            result,
            [
                {
                    "entrance": self.entrances[0],
                    "timestamp": self.base_time + timedelta(minutes=i),
                    "system_entry": i < 3,
                    "manual_entry": i < 3,
                    "is_equal": True,
                }
                for i in range(6)
            ],
        )

    def test_all_opposite(self) -> None:
        for i in range(3):
            TestExitEvent.objects.create(
                entrance=self.entrances[0],
                timestamp=self.base_time + timedelta(minutes=i),
            )
        for i in range(3, 6):
            TestEntryEvent.objects.create(
                entrance=self.entrances[0],
                timestamp=self.base_time + timedelta(minutes=i),
            )

        result = self._get_mapped()
        self.assertSequenceEqual(
            result,
            [
                {
                    "entrance": self.entrances[0],
                    "timestamp": self.base_time + timedelta(minutes=i),
                    "system_entry": i < 3,
                    "manual_entry": i >= 3,
                    "is_equal": False,
                }
                for i in range(6)
            ],
        )

    def test_no_manual(self) -> None:
        result = self._get_mapped()
        self.assertSequenceEqual(
            result,
            [
                {
                    "entrance": self.entrances[0],
                    "timestamp": self.base_time + timedelta(minutes=i),
                    "system_entry": i < 3,
                    "manual_entry": None,
                    "is_equal": False,
                }
                for i in range(6)
            ],
        )

    def test_no_system(self) -> None:
        EntryEvent.objects.all().delete()
        ExitEvent.objects.all().delete()

        for i in range(3):
            TestExitEvent.objects.create(
                entrance=self.entrances[0],
                timestamp=self.base_time + timedelta(minutes=i),
            )
        for i in range(3, 6):
            TestEntryEvent.objects.create(
                entrance=self.entrances[0],
                timestamp=self.base_time + timedelta(minutes=i),
            )

        result = self._get_mapped()
        self.assertSequenceEqual(
            result,
            [
                {
                    "entrance": self.entrances[0],
                    "timestamp": self.base_time + timedelta(minutes=i),
                    "system_entry": None,
                    "manual_entry": i >= 3,
                    "is_equal": False,
                }
                for i in range(6)
            ],
        )

    def test_wrong_entrance(self) -> None:
        for i in range(3):
            TestEntryEvent.objects.create(
                entrance=self.entrances[1],
                timestamp=self.base_time + timedelta(minutes=i),
            )
        for i in range(3, 6):
            TestExitEvent.objects.create(
                entrance=self.entrances[1],
                timestamp=self.base_time + timedelta(minutes=i),
            )

        result = self._get_mapped()
        self.assertEqual(len(result), 12)
        self.maxDiff = None
        expected = [
            {
                "entrance": self.entrances[0],
                "timestamp": self.base_time + timedelta(minutes=i),
                "system_entry": i < 3,
                "manual_entry": None,
                "is_equal": False,
            }
            for i in range(6)
        ] + [
            {
                "entrance": self.entrances[1],
                "timestamp": self.base_time + timedelta(minutes=i),
                "system_entry": None,
                "manual_entry": i < 3,
                "is_equal": False,
            }
            for i in range(6)
        ]
        expected.sort(key=lambda x: x["timestamp"])
        self.assertSequenceEqual(result, expected)
