import csv
from datetime import datetime

from api.models import DeviceAPIKey
from components.test_event_results.test_event_results import get_test_results
from dashboard.utils import filter_events
from django.contrib.auth.decorators import permission_required
from django.core.cache import cache
from django.core.exceptions import PermissionDenied, ValidationError
from django.db.models import Max, Value
from django.db.models.functions import Coalesce, Greatest
from django.http import Http404, HttpRequest, HttpResponse, StreamingHttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views.decorators.http import require_http_methods
from occupancy.forms import ConfigureDeviceForm, FilterEventsForm
from occupancy.models import Device, Entrance, TestEntryEvent, TestExitEvent


def index(request: HttpRequest) -> HttpResponse:
    return redirect("dashboard")


def configuration(request: HttpRequest) -> HttpResponse:
    entrances = (
        Entrance.objects.annotate(
            latest_entry=Max("entries__timestamp"),
            latest_exit=Max("exits__timestamp"),
        )
        .annotate(
            latest_event=Greatest(
                Coalesce("latest_entry", Value(datetime.min)),
                Coalesce("latest_exit", Value(datetime.min)),
            )
        )
        .order_by("id")
    )
    return render(request, "configuration.html", {"entrances": entrances})


@permission_required(
    ("occupancy.view_testentryevent", "occupancy.view_testexitevent"),
    raise_exception=True,
)
def test_events(request: HttpRequest) -> HttpResponse:
    return render(request, "test_events.html")


@permission_required(
    ("occupancy.view_testentryevent", "occupancy.view_testexitevent"),
    raise_exception=True,
)
def test_event_export(request: HttpRequest) -> HttpResponse:
    test_results = get_test_results()
    pseudo_buffer = Echo()
    writer = csv.writer(pseudo_buffer)

    def rows_generator():
        yield writer.writerow(["Tid", "Indgang", "System", "Manuel", "Status"])
        for item in test_results:
            yield writer.writerow(
                [
                    item["timestamp"].isoformat(),
                    item["entrance"].name,
                    item["system_entry"],
                    item["manual_entry"],
                    item["is_equal"],
                ]
            )

    response = StreamingHttpResponse(
        (row for row in rows_generator()), content_type="text/csv"
    )
    today = timezone.localtime().date()
    response["Content-Disposition"] = (
        f'attachment; filename="innhabit_test_data_{today}.csv"'
    )
    return response


@permission_required(
    ("occupancy.add_testentryevent", "occupancy.add_testexitevent"),
    raise_exception=True,
)
def select_test_entrance(requests: HttpRequest) -> HttpResponse:
    entrances = Entrance.objects.all()
    return render(requests, "select_test_entrance.html", {"entrances": entrances})


@permission_required(
    ("occupancy.add_testentryevent", "occupancy.add_testexitevent"),
    raise_exception=True,
)
def add_test_events(request: HttpRequest, pk: int) -> HttpResponse:
    entrance = get_object_or_404(Entrance, pk=pk)
    if request.method == "POST":
        action = request.POST.get("action").casefold()
        match action:
            case "entry":
                model = TestEntryEvent
            case "exit":
                model = TestExitEvent
            case _:
                raise ValidationError("Invalid action")
        model.objects.create(entrance=entrance, timestamp=timezone.now())

    today = timezone.localtime().date()
    entries = TestEntryEvent.objects.filter(
        entrance=entrance, timestamp__date=today
    ).count()
    exits = TestExitEvent.objects.filter(
        entrance=entrance, timestamp__date=today
    ).count()
    occupancy = entries - exits
    context = {
        "entrance": entrance,
        "current_occupancy": occupancy,
    }
    if request.htmx:
        return HttpResponse(occupancy)
    return render(request, "add_test_events.html", context)


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
    settings = None
    if device:
        image_obj = device.images.order_by("-created_at").first()
        settings = device.settings
        if image_obj:
            image = image_obj.image

    form = None
    if request.method == "POST":
        if not device or not settings:
            raise Http404
        messages = set(cache.get(f"device-{device.pk}-messages", []))
        form = ConfigureDeviceForm(request.POST)
        if form.is_valid() and device:
            if (
                settings.entry_box != form.entry_box
                or settings.exit_box != form.exit_box
            ) and not form.cleaned_data["request_image"]:
                settings.entry_box = form.entry_box
                settings.exit_box = form.exit_box
                settings.save()
                messages.add("update_settings")
            if form.cleaned_data["request_image"]:
                messages.add("request_image")
            if form.entry_box or form.exit_box:
                messages.add("update_settings")
            cache.set(f"device-{device.pk}-messages", list(messages))

    allow_image_request = device and "request_image" not in cache.get(
        f"device-{device.pk}-messages", []
    )
    context = {
        "entrance": entrance,
        "device": device,
        "image": image,
        "api_key": api_key,
        "api_key_available": api_key_available,
        "allow_image_request": allow_image_request,
        "form": form,
        "settings": settings,
    }
    return render(request, "configure_entrance.html", context)


def export_view(request: HttpRequest) -> HttpResponse:
    context = {"entrances": Entrance.objects.order_by("name").all()}
    return render(request, "export.html", context)


class Echo:
    def write(self, value):
        return value


def export_data(request: HttpRequest) -> HttpResponse:
    filters = FilterEventsForm(request.GET)
    if not filters.is_valid():
        return HttpResponse(filters.errors.items(), status=400)

    events = filter_events(
        user=request.user,
        **filters.cleaned_data,
    )

    if not events.exists():
        raise Http404

    events = events.order_by("timestamp")

    pseudo_buffer = Echo()
    writer = csv.writer(pseudo_buffer)

    def rows_generator():
        yield writer.writerow(["Indgang", "Tidspunkt", "Retning"])
        for item in events.iterator(chunk_size=500):
            yield writer.writerow([item.entrance.name, item.timestamp, item.direction])

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
