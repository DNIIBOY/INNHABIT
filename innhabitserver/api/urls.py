from api.views import (
    DeviceImageViewset,
    EntranceViewset,
    EntryEventViewset,
    ExitEventViewset,
)
from rest_framework import routers

router = routers.SimpleRouter()
router.register("entrances", EntranceViewset)
router.register("events/entries", EntryEventViewset)
router.register("events/exits", ExitEventViewset)
router.register("images", DeviceImageViewset)

urlpatterns = router.urls
