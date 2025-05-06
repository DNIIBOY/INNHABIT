from datetime import UTC, datetime

from api.models import DeviceAPIKey
from django.contrib.auth import get_user_model
from django.urls import reverse
from django.utils import timezone
from occupancy.models import Device, Entrance, EntryEvent, ExitEvent
from rest_framework.test import APIClient, APITestCase

User = get_user_model()


class TestEntranceViewset(APITestCase):
    def setUp(self) -> None:
        self.client = APIClient()
        self.entrance = Entrance.objects.create(name="Entrance 1")
        self.user = User.objects.create_superuser(email="user@example.com")
        self.client.force_authenticate(self.user)

    def test_retrieve(self) -> None:
        url = reverse("entrance-detail", args=[self.entrance.id])
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["name"], "Entrance 1")

    def test_list(self) -> None:
        url = reverse("entrance-list")
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.data), 1)
        Entrance.objects.create(name="Entrance 2")
        response = self.client.get(url)
        self.assertEqual(len(response.data), 2)

    def test_delete(self) -> None:
        url = reverse("entrance-detail", args=[self.entrance.id])
        response = self.client.delete(url)
        self.assertEqual(response.status_code, 204)
        self.assertIsNone(response.data)

    def test_update(self) -> None:
        url = reverse("entrance-detail", args=[self.entrance.id])
        response = self.client.patch(url, {"name": "Test Entrance"}, format="json")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["name"], "Test Entrance")
        self.entrance.refresh_from_db()
        self.assertEqual(self.entrance.name, "Test Entrance")

    def test_create(self) -> None:
        self.client.logout()
        url = reverse("entrance-list")
        response = self.client.post(url, {"name": "Test Entrance"}, format="json")
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.data["name"], "Test Entrance")
        self.assertEqual(Entrance.objects.count(), 2)
        obj = Entrance.objects.get(id=response.data["id"])
        self.assertEqual(obj.name, "Test Entrance")


class TestEntryEventViewset(APITestCase):
    def setUp(self) -> None:
        self.test_class = EntryEvent
        self.view_base = "entryevent"
        self.client = APIClient()
        self.time = datetime(2003, 3, 3, 3, 3, 3, 30, tzinfo=UTC)
        self.entrance = Entrance.objects.create(name="Entrance 1")
        self.device = Device.objects.create(entrance=self.entrance)
        self.api_key = DeviceAPIKey.objects.create(device=self.device)
        self.api_key_headers = {"Authorization": f"Bearer {self.api_key}"}
        self.user = User.objects.create_superuser(email="user@example.com")
        self.client.force_authenticate(self.user)
        self.event = EntryEvent.objects.create(
            entrance=self.entrance, timestamp=self.time
        )

    def test_retrieve(self) -> None:
        url = reverse(self.view_base + "-detail", args=[self.event.id])
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(datetime.fromisoformat(response.data["timestamp"]), self.time)

    def test_list(self) -> None:
        url = reverse(self.view_base + "-list")
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.data), 1)
        self.test_class.objects.create(entrance=self.entrance, timestamp=self.time)
        response = self.client.get(url)
        self.assertEqual(len(response.data), 2)

    def test_delete(self) -> None:
        url = reverse(self.view_base + "-detail", args=[self.event.id])
        response = self.client.delete(url)
        self.assertEqual(response.status_code, 405)

    def test_update(self) -> None:
        url = reverse(self.view_base + "-detail", args=[self.event.id])
        response = self.client.patch(
            url, {"timestamp": datetime(2000, 1, 1, 0, 0, 0)}, format="json"
        )
        self.assertEqual(response.status_code, 405)
        self.event.refresh_from_db()
        self.assertEqual(self.event.timestamp, self.time)

    def test_create_invalid_time(self) -> None:
        self.client.logout()
        url = reverse(self.view_base + "-list")
        response = self.client.post(
            url,
            {"entrance": self.entrance.id, "timestamp": self.time},
            headers=self.api_key_headers,
            format="json",
        )
        self.assertEqual(response.status_code, 400)

    def test_create(self) -> None:
        self.client.logout()
        url = reverse(self.view_base + "-list")
        time = timezone.now()
        response = self.client.post(
            url,
            {"entrance": self.entrance.id, "timestamp": time},
            headers=self.api_key_headers,
            format="json",
        )
        self.assertEqual(response.status_code, 201)
        self.assertEqual(datetime.fromisoformat(response.data["timestamp"]), time)
        self.assertEqual(self.test_class.objects.count(), 2)
        obj = self.test_class.objects.get(id=response.data["id"])
        self.assertEqual(obj.timestamp, time)


class ExitEventViewset(TestEntryEventViewset):
    def setUp(self) -> None:
        super().setUp()
        self.test_class = ExitEvent
        self.view_base = "exitevent"
        self.event = ExitEvent.objects.create(
            entrance=self.entrance, timestamp=self.time
        )
