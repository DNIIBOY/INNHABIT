from django_components import Component, register


@register("latest_events")
class LatestEvents(Component):
    template_name = "latest_events.html"
