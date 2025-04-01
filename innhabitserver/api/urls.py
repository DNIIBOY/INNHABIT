from api.views import (
    DeviceImageViewset,
    DeviceSettingsView,
    EntranceViewset,
    EntryEventViewset,
    ExitEventViewset,
    device_poll,
)
from django.urls import path
from rest_framework import routers

router = routers.SimpleRouter()
router.register("entrances", EntranceViewset)
router.register("events/entries", EntryEventViewset)
router.register("events/exits", ExitEventViewset)
router.register("images", DeviceImageViewset)

urlpatterns = [
    path("device/poll/", device_poll),
    path("device/settings/", DeviceSettingsView.as_view()),
]

urlpatterns += router.urls
