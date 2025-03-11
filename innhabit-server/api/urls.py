from api.views import EntranceViewset, EntryEventViewset, ExitEventViewset
from rest_framework import routers

router = routers.SimpleRouter()
router.register("entrances", EntranceViewset)
router.register("events/entries", EntryEventViewset)
router.register("events/exits", ExitEventViewset)

urlpatterns = router.urls
