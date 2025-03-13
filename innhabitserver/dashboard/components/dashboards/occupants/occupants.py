from django_components import Component, register


@register("occupants")
class Occupants(Component):
    template_name = "occupants.html"
