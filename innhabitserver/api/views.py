from typing import Any

from api.serializers import (
    DeviceImageSerializer,
    DeviceSettingsSerializer,
    EntranceSerializer,
    EntryEventSerializer,
    ExitEventSerializer,
)
from django.core.cache import cache
from django.core.files.base import ContentFile
from django.db.models.query import QuerySet
from occupancy.models import (
    DeviceImage,
    DeviceSettings,
    Entrance,
    EntryEvent,
    ExitEvent,
)
from rest_framework import status, viewsets
from rest_framework.decorators import api_view, permission_classes
from rest_framework.exceptions import NotFound, PermissionDenied
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
from rest_framework.views import APIView

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

    def get_queryset(self) -> QuerySet[EntryEvent]:
        queryset = super().get_queryset()
        if isinstance(self.request.auth, DeviceAPIKey):
            raise PermissionDenied
        return queryset

    def create(self, request: Request, *args: Any, **kwargs: Any) -> Response:
        serializer = self.get_serializer(data=request.data.copy())
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
        serializer = self.get_serializer(data=request.data.copy())
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

    def perform_create(self, serializer: DeviceImageSerializer) -> None:
        device = serializer.validated_data["device"]
        messages = set(cache.get(f"device-{device.pk}-messages", []))
        messages.discard("request_image")
        messages = cache.set(f"device-{device.pk}-messages", list(messages))
        super().perform_create(serializer)


class DeviceSettingsView(APIView):
    permission_classes = [DeviceAPIKeyPermission]
    serializer_class = DeviceSettingsSerializer
    queryset = DeviceSettings.objects.all()

    def get(self, request: Request, *args: Any, **kwargs: Any) -> Response:
        instance = self.get_object()
        device = instance.device
        messages = set(cache.get(f"device-{device.pk}-messages", []))
        messages.discard("update_settings")
        messages = cache.set(f"device-{device.pk}-messages", list(messages))

        serializer = self.serializer_class(instance)
        return Response(serializer.data)

    def get_object(self) -> DeviceSettings:
        device = self.request.auth.device
        settings = getattr(device, "settings", None)
        if not settings:
            raise NotFound
        return settings


@api_view(["GET"])
@permission_classes([DeviceAPIKeyPermission])
def device_poll(request: Request) -> Response:
    device = request.auth.device
    messages = cache.get(f"device-{device.pk}-messages")
    if not messages:
        return Response(status=204)
    return Response(messages)
