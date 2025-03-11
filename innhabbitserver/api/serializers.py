from rest_framework import serializers
from occupancy.models import Entrance, EntryEvent, ExitEvent

class EntranceSerializer(serializers.ModelSerializer):
    class Meta:
        model = Entrance
        exclude = []


class EntryEventSerializer(serializers.ModelSerializer):
    class Meta:
        model = EntryEvent
        exclude = []
    
    
class ExitEventSerializer(serializers.ModelSerializer):
    class Meta:
        model = ExitEvent
        exclude = []
    
    