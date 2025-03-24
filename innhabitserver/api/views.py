from typing import Any

from api.serializers import (
    EntranceSerializer,
    EntryEventSerializer,
    ExitEventSerializer,
)
from occupancy.models import Entrance, EntryEvent, ExitEvent
from rest_framework import status, viewsets
from rest_framework.permissions import DjangoModelPermissions
from rest_framework.request import Request
from rest_framework.response import Response

from .models import DeviceAPIKey
from .permissions import DeviceAPIKeyPermission


class EntranceViewset(viewsets.ModelViewSet):
    queryset = Entrance.objects.all()
    serializer_class = EntranceSerializer


class EntryEventViewset(
    viewsets.ReadOnlyModelViewSet, viewsets.mixins.CreateModelMixin
):
    permission_classes = [DeviceAPIKeyPermission | DjangoModelPermissions]
    queryset = EntryEvent.objects.all()
    serializer_class = EntryEventSerializer

    def create(self, request: Request, *args: Any, **kwargs: Any) -> Response:
        serializer = self.get_serializer(data=request.data)
        if isinstance(request.auth, DeviceAPIKey):
            key = request.auth
            serializer.initial_data["entrance"] = key.device.entrance.pk
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        headers = self.get_success_headers(serializer.data)
        return Response(
            serializer.data, status=status.HTTP_201_CREATED, headers=headers
        )


class ExitEventViewset(viewsets.ReadOnlyModelViewSet, viewsets.mixins.CreateModelMixin):
    queryset = ExitEvent.objects.all()
    serializer_class = ExitEventSerializer
