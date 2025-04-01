from occupancy.models import (
    DeviceImage,
    DeviceSettings,
    Entrance,
    EntryEvent,
    ExitEvent,
)
from rest_framework import serializers


class EntranceSerializer(serializers.ModelSerializer):
    class Meta:
        model = Entrance
        exclude: list[str] = []


class EntryEventSerializer(serializers.ModelSerializer):
    class Meta:
        model = EntryEvent
        exclude: list[str] = []


class ExitEventSerializer(serializers.ModelSerializer):
    class Meta:
        model = ExitEvent
        exclude: list[str] = []


class DeviceImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = DeviceImage
        exclude: list[str] = []


class DeviceSettingsSerializer(serializers.ModelSerializer):
    class Meta:
        model = DeviceSettings
        exclude = ["id", "device"]
