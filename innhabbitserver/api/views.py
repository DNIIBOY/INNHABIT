from api.serializers import (
    EntranceSerializer,
    EntryEventSerializer,
    ExitEventSerializer,
)
from occupancy.models import Entrance, EntryEvent, ExitEvent
from rest_framework import viewsets


class EntranceViewset(viewsets.ModelViewSet):
    queryset = Entrance.objects.all()
    serializer_class = EntranceSerializer


class EntryEventViewset(
    viewsets.ReadOnlyModelViewSet, viewsets.mixins.CreateModelMixin
):
    queryset = EntryEvent.objects.all()
    serializer_class = EntryEventSerializer


class ExitEventViewset(viewsets.ReadOnlyModelViewSet, viewsets.mixins.CreateModelMixin):
    queryset = ExitEvent.objects.all()
    serializer_class = ExitEventSerializer
