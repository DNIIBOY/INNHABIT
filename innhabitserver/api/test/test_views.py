from django.urls import reverse
from occupancy.models import Entrance
from rest_framework.test import APIClient, APITestCase


class TestEntranceViewset(APITestCase):
    def setUp(self) -> None:
        self.client = APIClient()
        self.entrance = Entrance.objects.create(name="Entrance 1")

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
