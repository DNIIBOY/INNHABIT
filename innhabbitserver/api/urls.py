from rest_framework import routers
from django.urls import path
from api.views import EntranceViewset, EntryEventViewset, ExitEventViewset

router = routers.SimpleRouter()
router.register("entrances", EntranceViewset)
router.register("events/entries", EntryEventViewset)
router.register("events/exits", ExitEventViewset)
                
urlpatterns = router.urls