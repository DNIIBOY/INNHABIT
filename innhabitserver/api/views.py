from typing import Any

from api.serializers import (
    DeviceImageSerializer,
    EntranceSerializer,
    EntryEventSerializer,
    ExitEventSerializer,
)
from django.core.files.base import ContentFile
from occupancy.models import DeviceImage, Entrance, EntryEvent, ExitEvent
from rest_framework import status, viewsets
from rest_framework.parsers import (
    BaseParser,
    FileUploadParser,
    FormParser,
    JSONParser,
    MultiPartParser,
)
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


class ExitEventViewset(EntryEventViewset):
    queryset = ExitEvent.objects.all()
    serializer_class = ExitEventSerializer


class RawImageParser(BaseParser):
    media_type = "image/*"

    def parse(self, stream, media_type=None, parser_context=None):  # type: ignore
        return stream.read()


class DeviceImageViewset(
    viewsets.ReadOnlyModelViewSet, viewsets.mixins.CreateModelMixin
):
    permission_classes = [DeviceAPIKeyPermission | DjangoModelPermissions]
    queryset = DeviceImage.objects.all()
    serializer_class = DeviceImageSerializer
    parser_classes = [JSONParser, FormParser, MultiPartParser, FileUploadParser]

    def create(self, request: Request, *args: Any, **kwargs: Any) -> Response:
        serializer = self.get_serializer(data=request.data)
        if request.headers.get("Content-Type").startswith("image/"):
            content = ContentFile(
                request.data["file"].read(), name=request.data["file"].name
            )
            serializer.initial_data["image"] = content
        if isinstance(request.auth, DeviceAPIKey):
            serializer.initial_data["device"] = request.auth.device.pk
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        headers = self.get_success_headers(serializer.data)
        return Response(
            serializer.data, status=status.HTTP_201_CREATED, headers=headers
        )
