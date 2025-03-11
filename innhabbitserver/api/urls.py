from rest_framework import routers
from api.views import EntranceViewset, EntryEventViewset, ExitEventViewset

router = routers.SimpleRouter()
router.register("entrances", EntranceViewset)
router.register("events/entries", EntryEventViewset)
router.register("events/exits", ExitEventViewset)

urlpatterns = router.urls
