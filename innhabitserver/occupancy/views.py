import csv

from api.models import DeviceAPIKey
from dashboard.utils import filter_events
from django.core.exceptions import PermissionDenied, ValidationError
from django.db.models import F, Max
from django.db.models.functions import Greatest
from django.http import Http404, HttpRequest, HttpResponse, StreamingHttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views.decorators.http import require_http_methods
from occupancy.models import Device, Entrance


def index(request: HttpRequest) -> HttpResponse:
    return redirect("dashboard")


def configuration(request: HttpRequest) -> HttpResponse:
    entrances = (
        Entrance.objects.annotate(
            latest_entry=Max("entries__timestamp"),
            latest_exit=Max("exits__timestamp"),
            latest_event=Greatest(F("latest_entry"), F("latest_exit")),
        )
        .order_by("id")
        .all()
    )
    return render(request, "configuration.html", {"entrances": entrances})


def configure_entrance(
    request: HttpRequest, pk: int, api_key: str | None = None
) -> HttpResponse:
    api_key_available = bool(api_key)
    entrance = get_object_or_404(Entrance, pk=pk)
    device = getattr(entrance, "device", None)

    if device and not api_key:
        key_obj = getattr(device, "api_key", None)
        if key_obj:
            api_key = str(key_obj)

    image = None
    if device:
        image_obj = device.images.order_by("-created_at").first()
        if image_obj:
            image = image_obj.image

    context = {
        "entrance": entrance,
        "device": device,
        "image": image,
        "api_key": api_key,
        "api_key_available": api_key_available,
    }
    return render(request, "configure_entrance.html", context)


def export_view(request: HttpRequest) -> HttpResponse:
    context = {"entrances": Entrance.objects.order_by("name").all()}
    return render(request, "export.html", context)


class Echo:
    def write(self, value):
        return value


def export_data(request: HttpRequest) -> HttpResponse:
    event_type = request.GET.get("eventType") or None
    from_date = request.GET.get("from") or None
    to_date = request.GET.get("to") or None
    entrances = request.GET.getlist("entrances") or None
    if entrances:
        entrances = list(map(int, entrances))

    events = filter_events(
        user=request.user,
        entrances=entrances,
        event_type=event_type,
        from_date=from_date,
        to_date=to_date,
    )

    if not events.exists():
        raise Http404

    events = events.order_by("timestamp")

    pseudo_buffer = Echo()
    writer = csv.writer(pseudo_buffer)

    def rows_generator():
        yield writer.writerow(["Indgang", "Tidspunkt", "Retning"])
        for item in events.iterator(chunk_size=500):
            yield writer.writerow([item.entrance.name, item.timestamp, item.type])

    response = StreamingHttpResponse(
        (row for row in rows_generator()), content_type="text/csv"
    )
    today = timezone.localtime().date()
    response["Content-Disposition"] = (
        f'attachment; filename="innhabit_data_{today}.csv"'
    )
    return response


@require_http_methods(["POST", "DELETE"])
def api_key_view(request: HttpRequest, device_id: int) -> HttpResponse:
    device = get_object_or_404(Device, pk=device_id)
    api_key = None
    deleted = False
    if request.method == "DELETE":
        if not request.user.has_perm("api.delete_deviceapikey"):
            raise PermissionDenied
        if not hasattr(device, "api_key"):
            raise ValidationError("Device does not have an API key")
        device.api_key.delete()
        deleted = True
    if request.method == "POST":
        if not request.user.has_perm("api.add_deviceapikey"):
            raise PermissionDenied
        if hasattr(device, "api_key"):
            raise ValidationError("Device already has an API key")
        api_key = DeviceAPIKey.objects.create(device=device)

    if not request.htmx:
        return redirect("configure_entrance", pk=device.entrance.pk, api_key=api_key)

    available = bool(api_key)
    key_obj = getattr(device, "api_key", None)
    if key_obj and not api_key and not deleted:
        api_key = str(key_obj)

    context = {
        "entrance": device.entrance,
        "device": device,
        "api_key": api_key,
        "api_key_available": available,
    }
    return render(request, "configure_entrance.html#config_page", context)
