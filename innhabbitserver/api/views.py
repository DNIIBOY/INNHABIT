from rest_framework import viewsets
from occupancy.models import Entrance, EntryEvent, ExitEvent
from api.serializers import EntranceSerializer, EntryEventSerializer, ExitEventSerializer

class EntranceViewset(viewsets.ModelViewSet):
    queryset = Entrance.objects.all()
    serializer_class = EntranceSerializer

class EntryEventViewset(viewsets.ReadOnlyModelViewSet, viewsets.mixins.CreateModelMixin):
    queryset = EntryEvent.objects.all() 
    serializer_class = EntryEventSerializer

class ExitEventViewset(viewsets.ReadOnlyModelViewSet, viewsets.mixins.CreateModelMixin):
    queryset = ExitEvent.objects.all()
    serializer_class = ExitEventSerializer


    
