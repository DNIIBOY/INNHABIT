from django_components import Component, register
from occupancy.models import DeviceSettings


@register("setup_entryboxes")
class SetupEntryboxes(Component):
    template_name = "setup_entryboxes.html"
    js_file = "setup_entryboxes.js"

    def get_context_data(self, image_url: str, settings: DeviceSettings) -> dict:
        return {
            "image_url": image_url,
            "entry_box": settings.entry_box,
            "exit_box": settings.exit_box,
        }
